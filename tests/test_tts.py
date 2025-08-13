from src import tts

# Mock TTS engine to avoid real audio output during testing
class DummyEngine:
    def say(self, text):
        pass
    def runAndWait(self):
        pass
    def stop(self):
        pass

# Replace pyttsx3.init with the dummy version
tts.pyttsx3.init = lambda: DummyEngine()

def test_toggle_speech_starts_and_stops():
    tts._active = False

    # Should start speaking
    state_after_start = tts.toggle_speech("Hello world.")
    assert state_after_start is True

    tts._active = False

    # Should start speaking again
    state_after_start_again = tts.toggle_speech("Hello again.")
    assert state_after_start_again is True

    tts._active = False

    # Should start speaking a third time
    state_after_stop = tts.toggle_speech("Hello stop.")
    assert state_after_stop is True

    # Should stop speaking
    tts._active = True
    state_after_final_stop = tts.toggle_speech("doesn't matter")
    assert state_after_final_stop is False