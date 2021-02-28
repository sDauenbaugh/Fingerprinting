# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import subprocess


def bssid_scan():
    results = subprocess.check_output(["netsh", "wlan", "show", "network", "mode=BSSID"])
    results = results.decode('ascii')
    return results


if __name__ == '__main__':
    print(bssid_scan())

