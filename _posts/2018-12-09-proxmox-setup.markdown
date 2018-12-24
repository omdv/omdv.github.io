---
layout: post
title:  "Homelab server"
tags: [homelab, server, configuration]
description: My notes while building and setting up homelab server.
---

# Introduction

My notes for a process of building and setting up a homelab server for self-hosting various applications. This is not intended to be a complete and elaborate guide, but rather a list of references I used, as well as some lessons learned, scripts I found to be useful, etc.

# 1. Hardware

The [/r/homelab](https://www.reddit.com/r/homelab/) has great hardware guides both for off-the-shelf options, as well as for building a custom server. Some noteable off-the-shelf options are: combinations of NAS and small-factor PCs, like Intel NUCs or pre-built mini-tower HP, Dell, Levono servers. For a moment I was seriously considering the [NAS killer 2.0](https://www.reddit.com/r/JDM_WAAAT/comments/963cv4/nas_killer_20_the_murder_of_a_first_gen_core_i7/) from reddit user JDM_WAAAT, however the form-factor and relatively high power consumption was a deal-breaker.

I ended up basing my build around the iconic [Fractal Design Node 304 mini ITX case](https://pcpartpicker.com/list/tf3gHh). Intel Pentium G4560 had a pretty good power specification. Having the Nvidia GPU with CUDA support was also a requirement, as I am doing some projects or courses involving deep learning from time to time and in the past I used to spend quite a bit by renting GPU instances - managed to get a small refurbished GTX 650 from Zotac on Black Friday.

The build process was relatively straightforward. The motherboard PSU cable turned out to occupy most of the free space inside the Node 304. Currently I have only 1 SSD and 2 mirrored HDD drives, so I removed one of HDD bays and it fits OK. However I will likely be ordering the custom-length PSU cables to fit all 6 3.5" HDDs in the future. There are several companies who offer those, e.g. [CableMod](https://cablemod.com). I also had to replace the included 140mm fan by Noctua's one as Fractal's was quite noisy. Now the server sits right under my TV set in the living room and noone can hear that it is on.

# 2. Installing Host OS

I selected Proxmox with ZFS mirror (hence 2x 2TB HDDs to start with). Will also be using SSD cache for ZFS pool - for now I have one 2.5" SSD installed for this, but in the future it makes sense to switch to PCIe 2.0 SSD to keep all 3x bays and 6x HDDs.

This [guide](https://blog.evilgeniustech.com/proxmox-with-zfs-raidz-ssd-caching/) works great. Before the first `apt-get update` don't forget to get the GPG key via `wget -O- "http://download.proxmox.com/debian/key.asc" | apt-key add -`.

I am using LXC instead of VM, which works great oob. I am also trying to use unpriviliged containers whenever possible. The only potential complexity is if you want to share some folders on the host with containers.

Transfer ssh key to host: `cat ~/.ssh/id_rsa.pub | ssh root@192.168.0.xxx 'cat >> .ssh/authorized_keys'`

### 2a. Useful Proxmox snippets

- Find out ip addr of the container: `pct exec <ID> -- ip addr`
- List/download templates: `pveam available`
- Bind folder or edit /etc/pve/lxc/<ID>.conf: `pct set <ID> -mp0 /pool/storage,mp=/srv/storage`

### 2b. Unpriviliged Turnkey Linux Containers

There are plenty of templates based on Turnkey linux for most popular self-hosted services. The only issue I found is that Turnkey LXCs does not support unpriviliged option during creation. There is a method, however, to [solve it](https://forum.proxmox.com/threads/unprivileged-containers.26148/page-2). The description got it slightly wrong - you need to take backup _after_ you removed random and urandom. Otherwise it works great.

### 2c. Creating shared folders

For most of the services there is no benefit in keeping data in the mounted folder on the host. Situation gets even bit more complexed if you want to share the folder with the unpriviliged container. Instead of matching permissions exactly I just `chmod 777` for all shared folders.

# 3. Turnkey Torrent Server

Install and use the method above to convert to unpriviliged. `adduser <ID>` and then `smbpasswd -a <ID>` to create a new user and add it to samba. Use webmin to configure access to mounted folder or `vi /etc/samba/smb.conf`.

# 3a. OpenVPN

High level steps:
- create `tun` on the host
- share it with unpriviliged container
- install OpenVPN inside container
- setup connection by getting settings from provider, create pass.txt and append to .ovpn file.
- [start on boot](https://askubuntu.com/questions/464264/starting-openvpn-client-automatically-at-boot)
- [bug](Bug inside unpriviliged container: https://askubuntu.com/questions/747023/systemd-fails-to-start-openvpn-in-lxd-managed-16-04-container?newreg=752ad6b8c92b48b5bfc08b44f6185692) when trying to start on boot
- [use iptables](https://askubuntu.com/questions/37412/how-can-i-ensure-transmission-traffic-uses-a-vpn) to limit transmission only to VPN

# 4. Media server (Emby)

Just share folders. Also I disabled samba inside this LXC and use samba only on Torrent server.
```
sudo pct set 100 -mp0 /pool/movies,mp=/srv/storage/Movies && sudo pct set 100 -mp1 /pool/tvshows,mp=/srv/storage/TVShows
sudo pct set 101 -mp0 /pool/movies,mp=/srv/storage/Movies && sudo pct set 101 -mp1 /pool/tvshows,mp=/srv/storage/TVShows
```

# 5. Nextcloud

Ended up not mounting any host folders and leaving it isolated as is. Use Turnkey `confconsole` to get letsencrypt certificate for the domain.

# 6. Monitoring

I am using TIG (telegraf-influxdb-grafana) stack for monitoring at the host level. Grafana and influxdb run as LXC, while telegraf runs at host level to get access to system-level parameters.

List of monitoring related services with notes is below.

### 6a. InfluxDB

Influxdb runs inside unpriviliged Alpine LXC in a tiny container (1Gb space, 256Mb RAM). Just install using apk and add to [startup scripts](https://www.cyberciti.biz/faq/how-to-enable-and-start-services-on-alpine-linux/). I had an issue with service crashing, as the `/etc/network/interfaces` was missing the `lo loopback interface`. Don't forget to enable the HTTP binding in `/etc/influxdb/influxdb.conf`, otherwise I left configuration untouched. Also add to Proxmox level backup and auto-startup.

### 6b. Grafana

You can run a compiled binary on Alpine if you [install glibc](https://github.com/sgerrand/alpine-pkg-glibc). However then you have to create a service to run at boot time, add user, so on. So I decided to use familiar debian LXC instead. Installed using apt-get. Don't forget to enable systemctl service to start on boot.

### 6c. Telegraf

Just install directly on the host, move configuration file to directory on ZFS pool to enable backup and symlink to `/etc/telegraf/`. Point influxdb in configuration file to IP from 7a.

### 6d. Additional monitoring utilities

Some additional utilities to get parameters not available directly in telegraf:
- `hddtemp` for drive temperature monitoring
- [Proxmox external metrics](https://pve.proxmox.com/wiki/External_Metric_Server), influxdb has to have UDP enabled.
- `nvidia-smi` is available oob, see instructions below on GPU part

# 7. Installing GPU support / CUDA

Overall [this guide](https://medium.com/@MARatsimbazafy/journey-to-deep-learning-nvidia-gpu-passthrough-to-lxc-container-97d0bc474957) has a lot of good information, but needs few adjustments already.

First need to install on host. Just use [official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Get proper headers first. I ended up installing using `apt-get install -t stretch-backports nvidia-cuda-toolkit` from `deb http://httpredir.debian.org/debian stretch-backports main contrib non-free`. Just make sure to update `stretch` to correct debian/proxmox version.

Enable start at boot as per [Nvidia guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-verifications). Just create a file in `/etc/init.d` and add to startup scripts.