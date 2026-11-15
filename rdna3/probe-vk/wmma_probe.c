// Headless Vulkan compute harness for VK_KHR_cooperative_matrix 16x16x16 probes.
// No surface, no swapchain, no WSI extension: it cannot open a window.
//
//   wmma_probe props
//   wmma_probe run <spv> <N> <eA> <eC> <A.bin> <B.bin> <C.bin> <out_prefix> <reps> [<disasm.txt>]
//
// Each problem is 256 A, 256 B, 256 C elements (row-major 16x16); eA / eC are element sizes in bytes.
// N must be a multiple of 4 (4 subgroups of 32 lanes per workgroup). Output: <out_prefix>.<rep>.bin.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

static const uint32_t WANT_VENDOR = 0x1002, WANT_DEVICE = 0x744c;

static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;

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

static const char *ctype(VkComponentTypeKHR t) {
    switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
    case VK_COMPONENT_TYPE_SINT16_KHR: return "s16";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
    case VK_COMPONENT_TYPE_SINT64_KHR: return "s64";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
    case VK_COMPONENT_TYPE_UINT16_KHR: return "u16";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
    case VK_COMPONENT_TYPE_UINT64_KHR: return "u64";
    default: { static char buf[32]; snprintf(buf, sizeof buf, "type%d", (int)t); return buf; }
    }
}

static void init(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "wmma_probe", 1, NULL, 0, VK_API_VERSION_1_3};
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

    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties ssc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, &drv};
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cmp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR, &ssc};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &cmp};
    vkGetPhysicalDeviceProperties2(phys, &p2);
    printf("device: %s (vendor 0x%04x device 0x%04x) api %u.%u.%u\n", p2.properties.deviceName, p2.properties.vendorID,
           p2.properties.deviceID, VK_API_VERSION_MAJOR(p2.properties.apiVersion), VK_API_VERSION_MINOR(p2.properties.apiVersion),
           VK_API_VERSION_PATCH(p2.properties.apiVersion));
    printf("driver: %s / %s\n", drv.driverName, drv.driverInfo);
    printf("subgroup size range: %u..%u; coopmat supported stages 0x%x\n", ssc.minSubgroupSize, ssc.maxSubgroupSize,
           cmp.cooperativeMatrixSupportedStages);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties *qf = calloc(nq, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
    qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    free(qf);

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pex = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, NULL, VK_TRUE};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, &pex, VK_TRUE, VK_FALSE};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cmf};
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = VK_TRUE;
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

static void props(void) {
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR fn =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    uint32_t n = 0;
    CK(fn(phys, &n, NULL));
    VkCooperativeMatrixPropertiesKHR *p = calloc(n, sizeof *p);
    for (uint32_t i = 0; i < n; i++) p[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    CK(fn(phys, &n, p));
    printf("cooperative matrix properties: %u entries\n", n);
    printf("  M  N  K  A    B    C    Result  saturatingAccumulation  scope\n");
    for (uint32_t i = 0; i < n; i++)
        printf("  %u %u %u %-4s %-4s %-4s %-7s %u                       %s\n", p[i].MSize, p[i].NSize, p[i].KSize, ctype(p[i].AType),
               ctype(p[i].BType), ctype(p[i].CType), ctype(p[i].ResultType), p[i].saturatingAccumulation,
               p[i].scope == VK_SCOPE_SUBGROUP_KHR ? "subgroup" : p[i].scope == VK_SCOPE_WORKGROUP_KHR ? "workgroup" : "other");
    free(p);
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; size_t size; } Buf;

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
    FILE *f = fopen(path, "w");
    if (!f || !gp || !gr) { fprintf(stderr, "cannot dump internal representations\n"); if (f) fclose(f); return; }
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, NULL, pipe};
    uint32_t ne = 0;
    CK(gp(dev, &pi, &ne, NULL));
    VkPipelineExecutablePropertiesKHR *ep = calloc(ne, sizeof *ep);
    for (uint32_t i = 0; i < ne; i++) ep[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    CK(gp(dev, &pi, &ne, ep));
    for (uint32_t e = 0; e < ne; e++) {
        fprintf(f, "=== executable %u: %s (%s), subgroup size %u\n", e, ep[e].name, ep[e].description, ep[e].subgroupSize);
        VkPipelineExecutableInfoKHR xi = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, NULL, pipe, e};
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
    fclose(f);
}

static int run(int argc, char **argv) {
    if (argc < 10) { fprintf(stderr, "usage: run <spv> <N> <eA> <eC> <A> <B> <C> <out_prefix> <reps> [disasm]\n"); return 1; }
    const char *spv_path = argv[1];
    uint32_t N = (uint32_t)strtoul(argv[2], NULL, 0);
    size_t eA = strtoul(argv[3], NULL, 0), eC = strtoul(argv[4], NULL, 0);
    int reps = atoi(argv[9]);
    if (N % 4) { fprintf(stderr, "N must be a multiple of 4\n"); return 1; }
    size_t sa, sb, sc, sz;
    void *ha = load_file(argv[5], &sa), *hb = load_file(argv[6], &sb), *hc = load_file(argv[7], &sc);
    size_t wa = (size_t)N * 256 * eA, wc = (size_t)N * 256 * eC;
    if (sa != wa || sb != wa || sc != wc) { fprintf(stderr, "input sizes %zu %zu %zu, expected %zu %zu\n", sa, sb, sc, wa, wc); return 1; }
    if (2 * wa + 2 * wc > (size_t)900 << 20) { fprintf(stderr, "buffers would exceed 900 MiB\n"); return 1; }
    uint32_t *code = load_file(spv_path, &sz);

    Buf ba = make_buf(wa), bb = make_buf(wa), bc = make_buf(wc), bd = make_buf(wc);
    memcpy(ba.map, ha, wa);
    memcpy(bb.map, hb, wa);
    memcpy(bc.map, hc, wc);

    VkDescriptorSetLayoutBinding bind[4];
    for (int i = 0; i < 4; i++)
        bind[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 4, bind};
    VkDescriptorSetLayout dsl;
    CK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 0, NULL};
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sz, code};
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, NULL, 32};
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL,
                                        VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,
                                        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &rss,
                                         VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL},
                                        pl, VK_NULL_HANDLE, 0};
    VkPipeline pipe;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    if (argc > 10) dump_executables(pipe, argv[10]);

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
    VkDescriptorSet ds;
    CK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    Buf *bs[4] = {&ba, &bb, &bc, &bd};
    VkDescriptorBufferInfo dbi[4];
    VkWriteDescriptorSet wds[4];
    for (int i = 0; i < 4; i++) {
        dbi[i] = (VkDescriptorBufferInfo){bs[i]->buf, 0, VK_WHOLE_SIZE};
        wds[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, (uint32_t)i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi[i], NULL};
    }
    vkUpdateDescriptorSets(dev, 4, wds, 0, NULL);

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfam};
    VkCommandPool cp;
    CK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(dev, &fci, NULL, &fence));

    char path[4096];
    for (int r = 0; r < reps; r++) {
        memset(bd.map, 0xA5, wc);  // a poison pattern, so an unwritten output cannot pass as a result
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        CK(vkBeginCommandBuffer(cb, &bi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdDispatch(cb, N / 4, 1, 1);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
        CK(vkQueueSubmit(queue, 1, &si, fence));
        CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_C(20000000000)));
        CK(vkResetFences(dev, 1, &fence));
        CK(vkResetCommandBuffer(cb, 0));
        snprintf(path, sizeof path, "%s.%d.bin", argv[8], r);
        FILE *f = fopen(path, "wb");
        if (!f || fwrite(bd.map, 1, wc, f) != wc) { perror(path); return 1; }
        fclose(f);
    }
    printf("ran %s N=%u reps=%d\n", spv_path, N, reps);

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cp, NULL);
    vkDestroyDescriptorPool(dev, dp, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    for (int i = 0; i < 4; i++) free_buf(bs[i]);
    free(ha); free(hb); free(hc); free(code);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: wmma_probe props | run ...\n"); return 1; }
    init();
    int rc = 0;
    if (!strcmp(argv[1], "props")) props();
    else if (!strcmp(argv[1], "run")) rc = run(argc - 1, argv + 1);
    else { fprintf(stderr, "unknown command\n"); rc = 1; }
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    return rc;
}
