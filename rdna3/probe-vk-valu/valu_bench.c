// Headless Vulkan compute timing harness for shader (VALU) throughput on the RX 7900 XT (1002:744c, RADV).
// No surface, no swapchain, no WSI extension: it cannot open a window. Adapted from ../probe-vk/wmma_probe.c.
//
//   valu_bench info
//   valu_bench run <spv> <workgroups> <iters> <reps> <in.bin|zero:BYTES> <out_bytes> <aux.bin|zero:BYTES> <out.bin|-> [disasm.txt]
//
// Bindings: 0 = in (read), 1 = out (write), 2 = aux (read: tables / weights). Push constant: uint iters.
// Workgroup size is whatever the shader declares; required subgroup size 32, full subgroups.
// Timing: two GPU timestamps (TOP and BOTTOM of pipe) around one dispatch per rep, after three untimed warm-up dispatches.
// Prints "rep <i> gpu_ns <ns>" per rep, then pipeline statistics.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

static const uint32_t WANT_VENDOR = 0x1002, WANT_DEVICE = 0x744c;
static const size_t VRAM_CAP = (size_t)900 << 20;

static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;
static float ts_period;

static void *load_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc(*size ? *size : 1);
    if (fread(p, 1, *size, f) != *size) { perror(path); exit(1); }
    fclose(f);
    return p;
}

// "zero:BYTES" gives a zero-filled block; anything else is a file.
static void *load_or_zero(const char *arg, size_t *size) {
    if (!strncmp(arg, "zero:", 5)) {
        *size = strtoull(arg + 5, NULL, 0);
        return calloc(1, *size ? *size : 4);
    }
    return load_file(arg, size);
}

static void init(int verbose) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "valu_bench", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 0;
    CK(vkEnumeratePhysicalDevices(inst, &n, NULL));
    VkPhysicalDevice *pds = calloc(n, sizeof *pds);
    CK(vkEnumeratePhysicalDevices(inst, &n, pds));
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.vendorID == WANT_VENDOR && p.deviceID == WANT_DEVICE) { phys = pds[i]; break; }
    }
    free(pds);
    if (!phys) { fprintf(stderr, "device %04x:%04x not found\n", WANT_VENDOR, WANT_DEVICE); exit(2); }

    VkPhysicalDeviceShaderCoreProperties2AMD cp2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD};
    VkPhysicalDeviceShaderCorePropertiesAMD cp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD, &cp2};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &cp};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(phys, &p2);
    ts_period = p2.properties.limits.timestampPeriod;
    if (verbose) {
        printf("device: %s (vendor 0x%04x device 0x%04x)\n", p2.properties.deviceName, p2.properties.vendorID, p2.properties.deviceID);
        printf("driver: %s / %s\n", drv.driverName, drv.driverInfo);
        printf("timestampPeriod_ns: %g\n", ts_period);
        printf("shaderEngineCount: %u\nshaderArraysPerEngineCount: %u\ncomputeUnitsPerShaderArray: %u\nsimdPerComputeUnit: %u\n"
               "wavefrontsPerSimd: %u\nwavefrontSize: %u\nvgprsPerSimd: %u\nvgprAllocationGranularity: %u\nactiveComputeUnitCount: %u\n",
               cp.shaderEngineCount, cp.shaderArraysPerEngineCount, cp.computeUnitsPerShaderArray, cp.simdPerComputeUnit,
               cp.wavefrontsPerSimd, cp.wavefrontSize, cp.vgprsPerSimd, cp.vgprAllocationGranularity, cp2.activeComputeUnitCount);
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties *qf = calloc(nq, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
    qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits) { qfam = i; break; }
    free(qf);
    if (qfam == UINT32_MAX) { fprintf(stderr, "no compute queue with timestamps\n"); exit(1); }

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pex = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, NULL, VK_TRUE};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, &pex, VK_TRUE, VK_FALSE};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cmf};
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = VK_TRUE;
    f13.shaderIntegerDotProduct = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.vulkanMemoryModel = VK_TRUE;
    f12.vulkanMemoryModelDeviceScope = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    f2.features.shaderInt16 = VK_TRUE;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfam, 1, &prio};
    const char *exts[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci, 0, NULL, 2, exts, NULL};
    CK(vkCreateDevice(phys, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; size_t size; } Buf;

// Device-local memory first (the timed reads of aux / in should come from VRAM, not system memory), host-visible so the
// harness can fill and read it directly (resizable BAR).
static Buf make_buf(size_t size) {
    Buf b = {0};
    b.size = size;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    CK(vkCreateBuffer(dev, &bci, NULL, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pick = UINT32_MAX;
    for (int pass = 0; pass < 2 && pick == UINT32_MAX; pass++)
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            uint32_t fl = mp.memoryTypes[i].propertyFlags;
            uint32_t need = want | (pass == 0 ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0);
            if ((mr.memoryTypeBits & (1u << i)) && (fl & need) == need) { pick = i; break; }
        }
    if (pick == UINT32_MAX) { fprintf(stderr, "no host-visible memory type\n"); exit(1); }
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, pick};
    CK(vkAllocateMemory(dev, &mai, NULL, &b.mem));
    CK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    CK(vkMapMemory(dev, b.mem, 0, size, 0, &b.map));
    return b;
}

static void free_buf(Buf *b) {
    vkUnmapMemory(dev, b->mem);
    vkDestroyBuffer(dev, b->buf, NULL);
    vkFreeMemory(dev, b->mem, NULL);
}

static void dump_executables(VkPipeline pipe, const char *path) {
    PFN_vkGetPipelineExecutablePropertiesKHR gp = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR gr =
        (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR gs = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableStatisticsKHR");
    FILE *f = path ? fopen(path, "w") : NULL;
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, NULL, pipe};
    uint32_t ne = 0;
    CK(gp(dev, &pi, &ne, NULL));
    VkPipelineExecutablePropertiesKHR *ep = calloc(ne, sizeof *ep);
    for (uint32_t i = 0; i < ne; i++) ep[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    CK(gp(dev, &pi, &ne, ep));
    for (uint32_t e = 0; e < ne; e++) {
        printf("subgroup_size %u\n", ep[e].subgroupSize);
        VkPipelineExecutableInfoKHR xi = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, NULL, pipe, e};
        uint32_t ns = 0;
        CK(gs(dev, &xi, &ns, NULL));
        VkPipelineExecutableStatisticKHR *st = calloc(ns, sizeof *st);
        for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        CK(gs(dev, &xi, &ns, st));
        for (uint32_t i = 0; i < ns; i++) {
            printf("stat %s: ", st[i].name);
            switch (st[i].format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: printf("%u\n", st[i].value.b32); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: printf("%lld\n", (long long)st[i].value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: printf("%llu\n", (unsigned long long)st[i].value.u64); break;
            default: printf("%g\n", st[i].value.f64);
            }
        }
        free(st);
        if (!f) continue;
        fprintf(f, "=== executable %u: %s (%s), subgroup size %u\n", e, ep[e].name, ep[e].description, ep[e].subgroupSize);
        uint32_t nr = 0;
        CK(gr(dev, &xi, &nr, NULL));
        VkPipelineExecutableInternalRepresentationKHR *ir = calloc(nr, sizeof *ir);
        for (uint32_t i = 0; i < nr; i++) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        CK(gr(dev, &xi, &nr, ir));
        for (uint32_t i = 0; i < nr; i++) ir[i].pData = calloc(1, ir[i].dataSize + 1);
        CK(gr(dev, &xi, &nr, ir));
        for (uint32_t i = 0; i < nr; i++) {
            fprintf(f, "--- %s: %s\n", ir[i].name, ir[i].description);
            if (ir[i].isText) fprintf(f, "%s\n", (char *)ir[i].pData);
            free(ir[i].pData);
        }
        free(ir);
    }
    free(ep);
    if (f) fclose(f);
}

static int run(int argc, char **argv) {
    if (argc < 9) { fprintf(stderr, "usage: run <spv> <wgs> <iters> <reps> <in|zero:N> <out_bytes> <aux|zero:N> <out.bin|-> [disasm]\n"); return 1; }
    const char *spv_path = argv[1];
    uint32_t wgs = (uint32_t)strtoul(argv[2], NULL, 0);
    uint32_t iters = (uint32_t)strtoul(argv[3], NULL, 0);
    int reps = atoi(argv[4]);
    size_t sin, saux, sz, sout = strtoull(argv[6], NULL, 0);
    void *hin = load_or_zero(argv[5], &sin), *haux = load_or_zero(argv[7], &saux);
    if (sin + sout + saux > VRAM_CAP) { fprintf(stderr, "buffers would exceed 900 MiB\n"); return 1; }
    uint32_t *code = load_file(spv_path, &sz);

    Buf bin = make_buf(sin), bout = make_buf(sout), baux = make_buf(saux);
    memcpy(bin.map, hin, sin);
    memcpy(baux.map, haux, saux);

    VkDescriptorSetLayoutBinding bind[3];
    for (int i = 0; i < 3; i++)
        bind[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, bind};
    VkDescriptorSetLayout dsl;
    CK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr};
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sz, code};
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, NULL, 32};
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL,
                                        VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR,
                                        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &rss,
                                         VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL},
                                        pl, VK_NULL_HANDLE, 0};
    VkPipeline pipe;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    dump_executables(pipe, argc > 9 ? argv[9] : NULL);

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
    VkDescriptorSet ds;
    CK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    Buf *bs[3] = {&bin, &bout, &baux};
    VkDescriptorBufferInfo dbi[3];
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        dbi[i] = (VkDescriptorBufferInfo){bs[i]->buf, 0, VK_WHOLE_SIZE};
        wds[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, (uint32_t)i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi[i], NULL};
    }
    vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);

    VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2, 0};
    VkQueryPool qp;
    CK(vkCreateQueryPool(dev, &qpci, NULL, &qp));
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfam};
    VkCommandPool cpool;
    CK(vkCreateCommandPool(dev, &cpi, NULL, &cpool));
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(dev, &fci, NULL, &fence));

    uint32_t pc[4] = {iters, 0, 0, 0};
    for (int r = -3; r < reps; r++) {   // r < 0: three untimed warm-ups (clocks ramp, caches and TLB warm)
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        CK(vkBeginCommandBuffer(cb, &bi));
        vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        vkCmdDispatch(cb, wgs, 1, 1);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
        CK(vkQueueSubmit(queue, 1, &si, fence));
        CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_C(20000000000)));
        CK(vkResetFences(dev, 1, &fence));
        CK(vkResetCommandBuffer(cb, 0));
        uint64_t ts[2];
        CK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, sizeof ts[0], VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        if (r >= 0) printf("rep %d gpu_ns %.0f\n", r, (double)(ts[1] - ts[0]) * ts_period);
    }
    if (strcmp(argv[8], "-")) {
        FILE *f = fopen(argv[8], "wb");
        if (!f || fwrite(bout.map, 1, sout, f) != sout) { perror(argv[8]); return 1; }
        fclose(f);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cpool, NULL);
    vkDestroyQueryPool(dev, qp, NULL);
    vkDestroyDescriptorPool(dev, dp, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    for (int i = 0; i < 3; i++) free_buf(bs[i]);
    free(hin); free(haux); free(code);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: valu_bench info | run ...\n"); return 1; }
    init(!strcmp(argv[1], "info"));
    int rc = 0;
    if (!strcmp(argv[1], "info")) rc = 0;
    else if (!strcmp(argv[1], "run")) rc = run(argc - 1, argv + 1);
    else { fprintf(stderr, "unknown command\n"); rc = 1; }
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    return rc;
}
