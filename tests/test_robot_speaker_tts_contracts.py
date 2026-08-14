from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_repo_text(path: str) -> str:
    return (REPO_ROOT / path).read_text()


class RobotSpeakerTtsContractsTest(unittest.TestCase):
    def test_robot_speaker_test_script_is_available(self):
        script = REPO_ROOT / "gear_sonic/scripts/test_g1_robot_speaker_tts.py"
        self.assertTrue(script.exists())
        source = script.read_text()

        for token in [
            "TestG1RobotSpeakerTtsConfig",
            "TextToSpeech",
            "detect_unitree_network_interface",
            "network_interface",
            "speaker_id",
            "post_speak_wait_s",
            "led_test",
            "LedControl",
            "beep_test",
            "PlayStream",
            "blocking=True",
            "time.sleep(max(0.0, config.post_speak_wait_s))",
        ]:
            self.assertIn(token, source)

    def test_robot_speaker_tts_initializes_unitree_audio_client(self):
        from gear_sonic.utils.data_collection.text_to_speech import RobotSpeakerTextToSpeech

        calls = []

        class FakeAudioClient:
            def GetVolume(self):
                calls.append(("AudioGetVolume",))
                return 0, {"volume": 42}

            def SetTimeout(self, timeout):
                calls.append(("SetTimeout", timeout))

            def Init(self):
                calls.append(("Init",))

            def SetVolume(self, volume):
                calls.append(("SetVolume", volume))
                return 0

            def TtsMaker(self, text, speaker_id):
                calls.append(("TtsMaker", text, speaker_id))
                return 0

        class FakeVuiClient:
            def SetTimeout(self, timeout):
                calls.append(("VuiSetTimeout", timeout))

            def Init(self):
                calls.append(("VuiInit",))

            def GetSwitch(self):
                calls.append(("VuiGetSwitch",))
                return 0, 0

            def SetSwitch(self, enable):
                calls.append(("VuiSetSwitch", enable))
                return 0

            def GetVolume(self):
                calls.append(("VuiGetVolume",))
                return 0, 55

            def SetVolume(self, level):
                calls.append(("VuiSetVolume", level))
                return 0

        def fake_channel_initialize(domain_id, network_interface=None):
            calls.append(("ChannelFactoryInitialize", domain_id, network_interface))
            return True

        tts = RobotSpeakerTextToSpeech(
            network_interface="eth0",
            volume=90,
            speaker_id=1,
            channel_initialize=fake_channel_initialize,
            audio_client_factory=FakeAudioClient,
            vui_client_factory=FakeVuiClient,
            enable_vui=True,
        )

        tts.print_and_say("Started recording", blocking=True)

        self.assertIn(("ChannelFactoryInitialize", 0, "eth0"), calls)
        self.assertIn(("VuiSetTimeout", 10.0), calls)
        self.assertIn(("VuiInit",), calls)
        self.assertEqual(calls.count(("VuiGetSwitch",)), 2)
        self.assertIn(("VuiSetSwitch", 1), calls)
        self.assertEqual(calls.count(("VuiGetVolume",)), 2)
        self.assertIn(("VuiSetVolume", 90), calls)
        self.assertIn(("SetTimeout", 10.0), calls)
        self.assertIn(("Init",), calls)
        self.assertEqual(calls.count(("AudioGetVolume",)), 2)
        self.assertIn(("SetVolume", 90), calls)
        self.assertIn(("TtsMaker", "Started recording", 1), calls)
        self.assertEqual(tts.last_return_code, 0)
        self.assertEqual(tts.initial_volume, (0, {"volume": 42}))
        self.assertEqual(tts.current_volume, (0, {"volume": 42}))
        self.assertEqual(tts.set_volume_return_code, 0)
        self.assertEqual(tts.vui_initial_switch, (0, 0))
        self.assertEqual(tts.vui_current_switch, (0, 0))
        self.assertEqual(tts.vui_set_switch_return_code, 0)
        self.assertEqual(tts.vui_initial_volume, (0, 55))
        self.assertEqual(tts.vui_current_volume, (0, 55))
        self.assertEqual(tts.vui_set_volume_return_code, 0)

    def test_bundled_unitree_audio_client_increments_tts_index(self):
        source = read_repo_text(
            "external_dependencies/unitree_sdk2_python/"
            "unitree_sdk2py/g1/audio/g1_audio_client.py"
        )

        self.assertNotIn("self.tts_index += self.tts_index", source)
        self.assertIn('p["index"] = self.tts_index', source)
        self.assertIn("self.tts_index += 1", source)

    def test_bundled_unitree_audio_client_supports_play_stream(self):
        client_base = read_repo_text(
            "external_dependencies/unitree_sdk2_python/unitree_sdk2py/rpc/client_base.py"
        )
        client = read_repo_text(
            "external_dependencies/unitree_sdk2_python/unitree_sdk2py/rpc/client.py"
        )
        audio_client = read_repo_text(
            "external_dependencies/unitree_sdk2_python/"
            "unitree_sdk2py/g1/audio/g1_audio_client.py"
        )

        self.assertIn("_CallBinaryWithParameterBase", client_base)
        self.assertIn("_CallBinaryWithParameter", client)
        self.assertIn("def PlayStream", audio_client)
        self.assertIn("ROBOT_API_ID_AUDIO_START_PLAY", audio_client)
        self.assertIn("def PlayStop", audio_client)
        self.assertIn("ROBOT_API_ID_AUDIO_STOP_PLAY", audio_client)

    def test_robot_speaker_tts_missing_sdk_does_not_raise(self):
        from gear_sonic.utils.data_collection.text_to_speech import RobotSpeakerTextToSpeech

        def fail_imports():
            raise ModuleNotFoundError("cyclonedds")

        tts = RobotSpeakerTextToSpeech(sdk_loader=fail_imports)
        tts.print_and_say("Started recording", blocking=True)

    def test_composite_tts_uses_priority_queue(self):
        from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech

        spoken = []

        class FakeBackend:
            def say(self, message, blocking=False):
                self.assert_blocking = blocking
                spoken.append(message)

            def wait_for_completion(self):
                pass

        backend = FakeBackend()
        tts = TextToSpeech(
            backends=[backend],
            autostart=False,
            robot_playback_seconds_per_character=0.0,
        )
        tts.say("保存完成", priority="status")
        tts.say("开始记录", priority="recording")
        tts.say("已丢弃", priority="critical")
        tts._start_worker()
        tts.wait_for_completion()

        self.assertEqual(spoken, ["已丢弃", "开始记录", "保存完成"])
        self.assertTrue(backend.assert_blocking)

    def test_robot_tts_retries_transient_exception_without_disabling(self):
        from gear_sonic.utils.data_collection.text_to_speech import RobotSpeakerTextToSpeech

        calls = []

        class FlakyAudioClient:
            def SetTimeout(self, _timeout):
                pass

            def Init(self):
                pass

            def SetVolume(self, _volume):
                return 0

            def TtsMaker(self, text, speaker_id):
                calls.append((text, speaker_id))
                if len(calls) == 1:
                    raise RuntimeError("temporary overload")
                return 0

        tts = RobotSpeakerTextToSpeech(
            channel_initialize=lambda *_args: True,
            audio_client_factory=FlakyAudioClient,
            vui_client_factory=None,
        )
        tts.say("开始记录", blocking=True)

        self.assertEqual(len(calls), 2)
        self.assertTrue(tts.available)
        self.assertEqual(tts.last_return_code, 0)

    def test_data_exporter_defaults_to_robot_speaker_backend(self):
        exporter = read_repo_text("gear_sonic/scripts/run_data_exporter.py")
        launcher = read_repo_text("gear_sonic/scripts/launch_data_collection.py")

        for token in [
            'text_to_speech_backend: Literal["robot", "local", "both"] = "robot"',
            "robot_tts_network_interface",
            "robot_tts_volume",
            "robot_tts_speaker_id: int = 0",
            "TextToSpeech(",
            "backend=config.text_to_speech_backend",
        ]:
            self.assertIn(token, exporter)

        for token in [
            'text_to_speech_backend: Literal["robot", "local", "both"] = "robot"',
            "robot_tts_network_interface",
            "robot_tts_speaker_id: int = 0",
            "--text-to-speech-backend",
            "--robot-tts-network-interface",
            "--robot-tts-volume",
            "--robot-tts-speaker-id",
        ]:
            self.assertIn(token, launcher)

    def test_data_exporter_recording_voice_prompts_are_chinese(self):
        exporter = read_repo_text("gear_sonic/scripts/run_data_exporter.py")

        for token in [
            "开始记录第",
            "条数据",
            "停止记录，正在保存",
            "保存完成",
            "已丢弃本条数据",
        ]:
            self.assertIn(token, exporter)

        for old_prompt in [
            "Started recording",
            "Stopping recording, preparing to save",
            "Finished saving episode",
            "Discarded episode",
        ]:
            self.assertNotIn(old_prompt, exporter)


if __name__ == "__main__":
    unittest.main()
