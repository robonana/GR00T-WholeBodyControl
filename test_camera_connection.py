#!/usr/bin/env python3
"""
相机连接测试脚本
快速测试相机服务器连接状态
"""

try:
    from gear_sonic.camera.sensor_server import SensorClient
    import time

    def test_camera_connection(host='localhost', port=5555, timeout=10):
        print("=" * 60)
        print(f"相机连接测试: {host}:{port}")
        print("=" * 60)

        try:
            # 创建客户端
            client = SensorClient(server_ip=host, port=port)
            print("✅ SensorClient创建成功")

            # 尝试接收数据
            print(f"\n等待相机数据 (超时{timeout}秒)...")
            for i in range(timeout):
                data = client.read(blocking=False)
                if data and data.get("images"):
                    camera_names = list(data["images"].keys())
                    print(f"\n✅ 成功连接到相机服务器！")
                    print(f"可用相机: {camera_names}")

                    # 显示每个相机的信息
                    for name, img in data["images"].items():
                        if img is not None:
                            h, w = img.shape[:2]
                            print(f"  - {name}: {w}x{h}px")

                    break
                time.sleep(1)
            else:
                print(f"\n❌ 超时: {timeout}秒内未收到相机数据")
                print("\n可能原因:")
                print("  1. 相机服务器未启动")
                print("  2. 主机地址或端口错误")
                print("  3. 网络连接问题")
                print("  4. 防火墙阻止连接")
                return False

            client.close()
            print("\n✅ 测试完成")
            return True

        except Exception as e:
            print(f"\n❌ 连接错误: {e}")
            print("\n请检查:")
            print("  1. 是否已启动相机服务器")
            print("  2. 网络连接是否正常")
            print("  3. 端口是否正确")
            return False

    if __name__ == "__main__":
        import sys
        host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5555

        test_camera_connection(host, port)

except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("\n请确保:")
    print("  1. 已激活正确的虚拟环境: source .venv_teleop/bin/activate")
    print("  2. 或安装相机相关依赖: pip install -e gear_sonic[teleop]")
