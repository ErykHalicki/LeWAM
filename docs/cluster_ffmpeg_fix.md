# torchcodec FFmpeg fix (cluster)

torchcodec requires FFmpeg shared libraries. The euler cluster has all FFmpeg 4 libs except `libavdevice.so.58`. Fix:

```bash
gcc -shared -fPIC -o .venv/lib/libavdevice.so.58 -x c /dev/null
echo 'export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> .venv/bin/activate
source .venv/bin/activate
```

This creates a stub `libavdevice.so.58` (torchcodec links against it but doesn't use device capture) and ensures both the stub and the real system FFmpeg 4 libs are on the library path.

Run once after setting up the venv.
