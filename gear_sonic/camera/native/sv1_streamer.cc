#include "unitree_mkp.h"

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <endian.h>
#include <fcntl.h>
#include <signal.h>
#include <string>
#include <unistd.h>

namespace {

volatile sig_atomic_t g_running = 1;
std::string g_device_path;

constexpr char kMagic[8] = {'S', 'V', '1', 'M', 'J', 'P', 'G', '1'};
constexpr uint32_t kProtocolVersion = 1;
constexpr uint32_t kMaximumFrameBytes = 16U * 1024U * 1024U;

#pragma pack(push, 1)
struct FrameHeader {
    char magic[8];
    uint32_t version_le;
    uint32_t payload_size_le;
    uint32_t sequence_le;
    uint32_t width_le;
    uint32_t height_le;
};
#pragma pack(pop)

static_assert(sizeof(FrameHeader) == 28, "unexpected SV1 frame header size");

void handle_signal(int) {
    g_running = 0;
}

bool install_signal_handler(int signal_number) {
    struct sigaction action {};
    action.sa_handler = handle_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;  // Let a blocking VIDIOC_DQBUF return EINTR.
    return sigaction(signal_number, &action, nullptr) == 0;
}

bool write_all(int fd, const void* data, size_t size) {
    const auto* cursor = static_cast<const uint8_t*>(data);
    while (size > 0) {
        const ssize_t written = write(fd, cursor, size);
        if (written > 0) {
            cursor += written;
            size -= static_cast<size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR && g_running) {
            continue;
        }
        return false;
    }
    return true;
}

bool parse_unsigned(const char* text, unsigned int* value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > UINT32_MAX) {
        return false;
    }
    *value = static_cast<unsigned int>(parsed);
    return true;
}

void usage(const char* program) {
    std::fprintf(
        stderr,
        "Usage: %s --device PATH [--camera-id 0|1] [--width N] "
        "[--height N] [--buffers N]\n",
        program);
}

}  // namespace

// The V2 SDK hard-codes CamId 0/1 to /dev/video0 and /dev/video1.  Redirect
// only that SDK open to the stable node selected by the caller.  Other opens
// (including the SDK's USB discovery) are passed through unchanged.
extern "C" int __real_open(const char* pathname, int flags, ...);

extern "C" int __wrap_open(const char* pathname, int flags, ...) {
    const char* actual_path = pathname;
    if (!g_device_path.empty() && pathname != nullptr &&
        std::strcmp(pathname, "/dev/video0") == 0) {
        actual_path = g_device_path.c_str();
    }

    if ((flags & O_CREAT) != 0) {
        va_list arguments;
        va_start(arguments, flags);
        const mode_t mode = static_cast<mode_t>(va_arg(arguments, int));
        va_end(arguments);
        return __real_open(actual_path, flags, mode);
    }
    return __real_open(actual_path, flags);
}

int main(int argc, char** argv) {
    unsigned int camera_id = 0;
    unsigned int width = 928;
    unsigned int height = 400;
    unsigned int buffer_count = 4;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            g_device_path = argv[++i];
        } else if (std::strcmp(argv[i], "--camera-id") == 0 && i + 1 < argc) {
            if (!parse_unsigned(argv[++i], &camera_id)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            if (!parse_unsigned(argv[++i], &width)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            if (!parse_unsigned(argv[++i], &height)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--buffers") == 0 && i + 1 < argc) {
            if (!parse_unsigned(argv[++i], &buffer_count)) {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (g_device_path.empty() || camera_id != 0 || width == 0 || height == 0 ||
        buffer_count < 2 || buffer_count > 32) {
        usage(argv[0]);
        return 2;
    }

    const int protocol_fd = dup(STDOUT_FILENO);
    if (protocol_fd < 0) {
        std::perror("dup(stdout)");
        return 3;
    }
    std::fflush(nullptr);
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        std::perror("redirect SDK stdout");
        close(protocol_fd);
        return 3;
    }

    signal(SIGPIPE, SIG_IGN);
    if (!install_signal_handler(SIGINT) || !install_signal_handler(SIGTERM)) {
        std::perror("sigaction");
        close(protocol_fd);
        return 3;
    }

    bool sdk_initialized = false;
    bool camera_initialized = false;
    bool buffers_allocated = false;
    bool stream_started = false;
    int exit_code = 0;

    std::fprintf(stderr, "[sv1_streamer] device=%s requested=%ux%u buffers=%u\n",
                 g_device_path.c_str(), width, height, buffer_count);

    if (UnitreeInitSdkEnv() != 0) {
        std::fprintf(stderr, "[sv1_streamer] UnitreeInitSdkEnv failed\n");
        exit_code = 4;
        goto cleanup;
    }
    sdk_initialized = true;

    // UnitreeInitCamera calls the SDK's internal InUnitreeOpenCamKey routine.
    // Calling public UnitreeOpenCamKey first sets a process-global flag that
    // makes UnitreeInitCamera reject streaming, so do not split these calls.
    if (UnitreeInitCamera(static_cast<int>(camera_id)) != 0) {
        std::fprintf(stderr, "[sv1_streamer] UnitreeInitCamera failed\n");
        exit_code = 5;
        goto cleanup;
    }
    camera_initialized = true;

    {
        ut_cam_param requested {};
        requested.width = width;
        requested.high = height;
        requested.pix = PIX_FMT_MJPEG;
        if (UnitreeSetCamParam(static_cast<int>(camera_id), &requested) != 0) {
            std::fprintf(stderr, "[sv1_streamer] UnitreeSetCamParam failed\n");
            exit_code = 6;
            goto cleanup;
        }
    }

    {
        ut_cam_param negotiated {};
        if (UnitreeGetCamParam(static_cast<int>(camera_id), &negotiated) != 0) {
            std::fprintf(stderr, "[sv1_streamer] UnitreeGetCamParam failed\n");
            exit_code = 7;
            goto cleanup;
        }
        width = negotiated.width;
        height = negotiated.high;
        if (negotiated.pix != PIX_FMT_MJPEG || width == 0 || height == 0) {
            std::fprintf(stderr, "[sv1_streamer] invalid negotiated camera mode\n");
            exit_code = 7;
            goto cleanup;
        }
    }

    if (UnitreeGetCamerbuffer(static_cast<int>(camera_id), buffer_count) != 0) {
        std::fprintf(stderr, "[sv1_streamer] UnitreeGetCamerbuffer failed\n");
        exit_code = 8;
        goto cleanup;
    }
    buffers_allocated = true;

    if (UnitreeStreamStart(static_cast<int>(camera_id)) != 0) {
        std::fprintf(stderr, "[sv1_streamer] UnitreeStreamStart failed\n");
        exit_code = 9;
        goto cleanup;
    }
    stream_started = true;
    std::fprintf(stderr, "[sv1_streamer] streaming negotiated MJPEG %ux%u\n",
                 width, height);

    for (uint32_t sequence = 0; g_running; ++sequence) {
        void* frame = nullptr;
        unsigned int index = 0;
        unsigned int bytes_used = 0;
        const int result = UnitreeGetOneFrame(
            static_cast<int>(camera_id), &frame, &index, &bytes_used);
        if (result != 0) {
            if (!g_running && errno == EINTR) {
                break;
            }
            std::fprintf(stderr, "[sv1_streamer] UnitreeGetOneFrame failed: %d\n", result);
            exit_code = 10;
            break;
        }

        bool valid = frame != nullptr && bytes_used >= 4 &&
                     bytes_used <= kMaximumFrameBytes;
        if (!valid) {
            std::fprintf(stderr, "[sv1_streamer] invalid frame size: %u\n", bytes_used);
            exit_code = 11;
        } else {
            FrameHeader header {};
            std::memcpy(header.magic, kMagic, sizeof(kMagic));
            header.version_le = htole32(kProtocolVersion);
            header.payload_size_le = htole32(bytes_used);
            header.sequence_le = htole32(sequence);
            header.width_le = htole32(width);
            header.height_le = htole32(height);
            if (!write_all(protocol_fd, &header, sizeof(header)) ||
                !write_all(protocol_fd, frame, bytes_used)) {
                if (errno != EPIPE) {
                    std::perror("[sv1_streamer] write frame");
                }
                g_running = 0;
            }
        }

        if (UnitreeReleaseOneFrame(static_cast<int>(camera_id), index) != 0) {
            std::fprintf(stderr, "[sv1_streamer] UnitreeReleaseOneFrame failed\n");
            exit_code = 12;
            break;
        }
        if (!valid) {
            break;
        }
    }

cleanup:
    if (stream_started) {
        UnitreeStreamStop(static_cast<int>(camera_id));
    }
    if (buffers_allocated) {
        UnitreeReleaseCamerBuffer(static_cast<int>(camera_id));
    }
    if (camera_initialized) {
        UnitreeDeInitCamera(static_cast<int>(camera_id));
    }
    if (sdk_initialized) {
        UnitreeDeInitSdkEnv();
    }
    close(protocol_fd);
    std::fprintf(stderr, "[sv1_streamer] stopped (exit=%d)\n", exit_code);
    return exit_code;
}
