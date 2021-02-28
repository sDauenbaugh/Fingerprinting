# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import subprocess


def bssid_scan():
    results = subprocess.check_output(["netsh", "wlan", "show", "network", "mode=BSSID"])
    results = results.decode('ascii')
    return results


def decode_network(network_string):
    lines = network_string.split('\n')
    if len(lines) < 6:
        return "not a network"
    ssid = lines[1].split(' : ')[1]
    bssid = lines[5].split(' : ')[1]
    signal = lines[6].split(' : ')[1]
    signal = int(signal[0:signal.index('%')])
    rssi = (signal / 2) - 100
    decoded_data = {'ssid': ssid[0:ssid.index('\r')], 'bssid': bssid[0:bssid.index('\r')],
                    'rssi': rssi}

    return decoded_data


if __name__ == '__main__':
    scan = bssid_scan()
    networks = scan.split('\n\r')
    networks.pop(0)
    networks.pop(len(networks)-1)
    decoded = []
    for n in networks:
        decoded.append(decode_network(n))
    for d in decoded:
        print(f'SSID: {d["ssid"]}')
        print(f'MAC: {d["bssid"]}')
        print(f'RSSI: {d["rssi"]}')
        print()
