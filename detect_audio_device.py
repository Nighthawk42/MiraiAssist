import pyaudio


def list_audio_devices():
    """Print a list of available audio devices with their indices and channel info."""
    pa = pyaudio.PyAudio()
    device_count = pa.get_device_count()

    for index in range(device_count):
        info = pa.get_device_info_by_index(index)
        name = info.get('name', '<Unknown>')
        max_input = info.get('maxInputChannels', 0)
        max_output = info.get('maxOutputChannels', 0)

        print(
            f"Device {index}: {name}\n"
            f"    Input Channels:  {max_input}\n"
            f"    Output Channels: {max_output}\n"
        )

    pa.terminate()


if __name__ == "__main__":
    list_audio_devices()