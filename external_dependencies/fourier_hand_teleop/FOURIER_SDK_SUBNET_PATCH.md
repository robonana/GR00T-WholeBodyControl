# Fourier SDK Subnet Patch Guide

This note records a real integration issue encountered while using Fourier
FDH-6 hands from `gear_sonic/scripts/pico_manager_thread_server.py` after the
hand IPs were moved from the default `192.168.137.x` subnet to
`192.168.123.x`.

It explains:

- what failed
- how the failure was diagnosed
- which code did **not** need changes
- which installed SDK library **did** need changes
- how to patch and roll back the change

## Symptom

The teleop manager started normally, but Fourier hand initialization failed:

```text
[FourierHandDriver] SDK init failed: Ret.FAIL. Attempt 1/5
...
[FourierHandDriver] Giving up on SDK init for now; will keep retrying discovery later.
```

At the same time, the hand IPs were reachable:

```bash
ping 192.168.123.19
ping 192.168.123.39
```

Both hands responded to ICMP, so basic L3 connectivity was not the issue.

## Root Cause

The failure was not caused by:

- `gear_sonic/scripts/pico_manager_thread_server.py`
- `gear_sonic/utils/teleop/solver/hand/fourier_hand_driver.py`
- left/right IP selection logic in this repository

The real cause was inside the installed Fourier SDK binary used by
`dexhandpy`.

`strace` on `DexHand().init()` showed that initialization sent broadcast
discovery packets to:

```text
192.168.137.255:2334
```

instead of the new subnet:

```text
192.168.123.255:2334
```

As a result, the SDK never discovered the devices on `192.168.123.x`, even
though direct ping worked.

## Important Distinction

### Repository code

The repository code already discovers devices dynamically:

- `fdh.get_ip_list()`
- `fdh.get_type(ip)`

See:

- `gear_sonic/utils/teleop/solver/hand/fourier_hand_driver.py`

That code does **not** hardcode `192.168.137.19` or `192.168.137.39`.

### Installed SDK library

The hardcoded subnet lives in the vendor SDK binary loaded by `dexhandpy`.

This means:

- changing repository Python code alone is not enough
- changing header files alone is not enough
- the loaded `.so` must be patched or rebuilt from real SDK source

## What Was Checked

### 1. Python package interface

The installed Python binding exposed only a compiled extension:

```text
dexhandpy/fdexhand.cpython-310-x86_64-linux-gnu.so
```

This is a wheel/binary install, not a full source checkout.

### 2. Header evidence

The SDK headers shipped with the package showed the default subnet constants:

```c
#define BROADCAST_ADDR "192.168.137.255"
#define LEFT_DEFAULT_ADDR "192.168.137.19"
#define RIGHT_DEFAULT_ADDR "192.168.137.39"
#define COMMUNICATION_PORT 2334
```

Example locations:

- `fdexhand/include/hand/fourierdexhand/basehand.h`
- `example/cpp/include/hand/fourierdexhand/basehand.h` from the SDK bundle

### 3. Actual runtime behavior

`strace` confirmed the loaded SDK was really broadcasting to
`192.168.137.255`, not `192.168.123.255`.

### 4. Actual loaded shared library

`ldd` on the Python extension showed the crucial detail:

the active `dexhandpy` extension loaded:

```text
libFourierDexHand.so.0
```

from a `uv` cache path, not from the obvious copy inside the virtual
environment.

This is why the first patch attempt changed the wrong file and had no effect.

## What Did Not Need Changes

No repository source change was required for subnet migration.

The following files were **not** the root cause for this subnet issue:

- `gear_sonic/scripts/pico_manager_thread_server.py`
- `gear_sonic/utils/teleop/solver/hand/fourier_hand_driver.py`

Those files can stay unchanged for this specific problem.

## What Actually Needed Changes

The installed Fourier SDK shared library needed a binary patch:

- replace `192.168.137.255`
- with `192.168.123.255`

### Why only the broadcast address was patched

For discovery, the only proven blocking constant was the broadcast address.

The strings:

- `192.168.137.19`
- `192.168.137.39`

did not appear in the actually loaded runtime library that was blocking
discovery, so there was no need to patch them for the observed failure.

## How to Find the Real Loaded Library

Always determine the actual loaded library first.

Example:

```bash
source .venv_teleop/bin/activate

python - <<'PY'
import importlib.util
spec = importlib.util.find_spec('dexhandpy.fdexhand')
print(spec.origin)
PY
```

Then inspect the extension dependencies:

```bash
ldd /path/to/dexhandpy/fdexhand*.so
readelf -d /path/to/dexhandpy/fdexhand*.so | egrep 'NEEDED|RPATH|RUNPATH'
```

In the observed environment, the active runtime library was:

```text
/home/wsy/data/.cache/uv/sdists-v9/pypi/dexhandpy/0.2.4/.../src/fdexhand/lib/libFourierDexHand.so.0
```

This path may differ across machines and environments.

## Reversible Patch Procedure

### 1. Verify the old string exists exactly once

```bash
python - <<'PY'
from pathlib import Path
path = Path('/path/to/libFourierDexHand.so.0')
data = path.read_bytes()
print(data.count(b'192.168.137.255'))
PY
```

Expected result:

```text
1
```

### 2. Create a backup and patch the library

```bash
python - <<'PY'
from pathlib import Path

src = Path('/path/to/libFourierDexHand.so.0')
bak = src.with_suffix(src.suffix + '.bak_192168137255')

old = b'192.168.137.255'
new = b'192.168.123.255'

data = src.read_bytes()
count = data.count(old)
print('match_count =', count)
if count != 1:
    raise SystemExit(f'Expected exactly 1 match, got {count}')

if not bak.exists():
    bak.write_bytes(data)
    print('backup_created =', bak)

patched = data.replace(old, new, 1)
src.write_bytes(patched)
print('patched =', src)
PY
```

### 3. Re-test the SDK directly

```bash
source .venv_teleop/bin/activate

python - <<'PY'
import dexhandpy.fdexhand as fdh
d = fdh.DexHand()
ret = d.init()
print('init ret =', ret)
if ret == fdh.Ret.SUCCESS:
    print('ip_list =', d.get_ip_list())
PY
```

Expected result after successful patch:

```text
init ret = Ret.SUCCESS
ip_list = ['192.168.123.19', '192.168.123.39']
```

### 4. Optional verification with `strace`

```bash
source .venv_teleop/bin/activate

strace -f -e trace=sendto,recvfrom python - <<'PY'
import dexhandpy.fdexhand as fdh
d = fdh.DexHand()
print(d.init())
PY
```

Expected evidence:

```text
sendto(..., inet_addr("192.168.123.255"), ...)
recvfrom(..., inet_addr("192.168.123.39"), ...)
recvfrom(..., inet_addr("192.168.123.19"), ...)
```

## Rollback

If the patch causes problems, restore the backup:

```bash
cp /path/to/libFourierDexHand.so.0.bak_192168137255 \
   /path/to/libFourierDexHand.so.0
```

Then re-run the minimal `dexhandpy` init test.

## Operational Notes

- This fix is **environment-local**, not repository-local.
- Reinstalling `dexhandpy`, rebuilding the virtual environment, or clearing the
  package cache may undo the patch.
- If you need this on multiple machines, either:
  - repeat the patch procedure
  - or obtain the real Fourier SDK source and rebuild properly

## Recommended Long-Term Fix

The proper long-term fix is not binary patching. It is one of:

- get the real SDK source for `libFourierDexHand.so.0`
- make broadcast subnet configurable
- rebuild `dexhandpy` / the Fourier SDK against the desired network settings

Binary patching is acceptable here only because:

- the old and new broadcast strings have equal length
- the failure was clearly isolated
- the patch is easily reversible

## Summary

For this subnet migration issue:

- repository Python code did **not** require modification
- the active Fourier SDK shared library did require modification
- the key patch was:

```text
192.168.137.255 -> 192.168.123.255
```

Once patched in the **actually loaded** `libFourierDexHand.so.0`, `dexhandpy`
successfully initialized and discovered:

- `192.168.123.19`
- `192.168.123.39`
