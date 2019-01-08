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

I ended up basing my build around the iconic [Fractal Design Node 304 mini ITX case](https://pcpartpicker.com/list/tf3gHh). Intel Pentium G4560 had a pretty good power specification. Having the Nvidia GPU with CUDA support was also a requirement, as I am doing some projects or courses involving deep learning from time to time and in the past I used to spend quite a bit by renting GPU instances - managed to get a really small refurbished GTX 650 from Zotac on Black Friday for about $50.

The build process was relatively straightforward. The motherboard PSU cable turned out to occupy most of the free space inside the Node 304. Currently I have only 1 SSD and 2 mirrored HDD drives, so I removed one of HDD bays and it fits OK. However I will likely be ordering the custom-length PSU cables to fit all 6 3.5" HDDs in the future. There are several companies who offer those, e.g. [CableMod](https://cablemod.com). I also had to replace the included 140mm fan by Noctua's one as Fractal's was quite noisy. Now the server sits right under my TV set in the living room and noone can hear that it is on.

# 2. Installing Host OS

I selected Proxmox with ZFS mirror (hence 2x 2TB HDDs to start with). Will also be using SSD cache for ZFS pool - for now I have one 2.5" SSD installed for this, but in the future it makes sense to switch to PCIe 2.0 SSD to keep all 3x bays and 6x HDDs.

This [guide](https://blog.evilgeniustech.com/proxmox-with-zfs-raidz-ssd-caching/) works great. Before the first `apt-get update` don't forget to get the GPG key via `wget -O- "http://download.proxmox.com/debian/key.asc" | apt-key add -`.

I am using LXC instead of VM, which works great oob. I am also trying to use unpriviliged containers whenever possible. The only potential complexity is if you want to share some folders on the host with containers.

Transfer ssh key to host: `cat ~/.ssh/id_rsa.pub | ssh root@192.168.0.xxx 'cat >> .ssh/authorized_keys'`

#### 2a. Useful Proxmox snippets

- Find out ip addr of the container: `pct exec <ID> -- ip addr`
- List/download templates: `pveam available`
- Bind folder or edit /etc/pve/lxc/<ID>.conf: `pct set <ID> -mp0 /pool/storage,mp=/srv/storage`
- Shrink size by backing up, restoring with e.g. `pct restore 101 /pool/dump/vzdump-<ID> --rootfs zfs-containers:4 --unprivileged 1`
- Assing static IP addresses to be equal to CT <ID>, i.e. start from 196.168.0.100/24
- I'll use 100-200 IP/ID range for constantly running apps, and 200+ for those launched on demand

#### 2b. Unpriviliged Turnkey Linux Containers

There are plenty of templates based on Turnkey linux for most popular self-hosted services. The only issue I found is that Turnkey LXCs does not support unpriviliged option during creation. There is a method, however, to [solve it](https://forum.proxmox.com/threads/unprivileged-containers.26148/page-2). The description got it slightly wrong - you need to take backup _after_ you removed random and urandom. Otherwise it works great.

#### 2c. Creating shared folders

For most of the services there is no benefit in keeping data in the mounted folder on the host. Situation gets even bit more complexed if you want to share the folder with the unpriviliged container. Instead of matching permissions exactly I changed permissions to `777` for all shared folders.

#### 2d. Setting up email and ZED

[Working guide](https://ubuntuforums.org/showthread.php?t=2404713&p=13811934#post13811934) for setting up zfs daemon for health notifications and also setting up email with external smtp. I used fastmail with app-specific password. After setting up zed use cron to schedule scrubs twice a month.

#### 2e. SMART tests

Use [smartd](https://wiki.archlinux.org/index.php/S.M.A.R.T.) to track smart attributes and schedule short/long tests. The actual service is actually `smartmontools`, [guide](https://help.ubuntu.com/community/Smartmontools).

Also use [telegraf plugin](https://github.com/influxdata/telegraf/tree/master/plugins/inputs/smart) to collect attributes and health info for dashboard. Give telegraf user sudo access to run just smartctl:
```
telegraf ALL=(ALL) NOPASSWD: /usr/sbin/smartctl
```

# 3. Reverse proxy server (ID 100)

Initially I set it up using Nginx Turnkey Linux container, but then I followed this alpine [guide](https://wiki.alpinelinux.org/wiki/Nginx_as_reverse_proxy_with_acme_(letsencrypt)) and it worked great. I did few changes however. The certificates are generated using `certbot` inside web hosting app containers, which are described below. The generated certificates are stored in the shared folder on the host. Renewal is handled by web containers as well. So nginx container is just using letsencrypt certificates and DH certificate. Sidenotes:
- mount the shared host `www` folder to `/mnt/www` as nginx creates its own in `/var/www`.
- when restoring container under different ID i had an issue with `/var/tmp/nginx` folder missing. Just create it with `nginx:nginx` owner and 700 permissions.

# 4. Turnkey Torrent Server (ID 101)

Install and use the method above to convert to unpriviliged. `adduser <ID>` and then `smbpasswd -a <ID>` to create a new user and add it to samba. Use webmin to configure access to mounted folder or `vi /etc/samba/smb.conf`.

I also configured openVPN inside container to limit transmission traffic only to VPN. High level steps:
- create `tun` on the host
- share it with unpriviliged container
- install OpenVPN inside container
- setup connection by getting settings from provider, create pass.txt and append to .ovpn file.
- [start on boot](https://askubuntu.com/questions/464264/starting-openvpn-client-automatically-at-boot)
- [bug](Bug inside unpriviliged container: https://askubuntu.com/questions/747023/systemd-fails-to-start-openvpn-in-lxd-managed-16-04-container?newreg=752ad6b8c92b48b5bfc08b44f6185692) when trying to start on boot
- [use iptables](https://askubuntu.com/questions/37412/how-can-i-ensure-transmission-traffic-uses-a-vpn) to limit transmission only to VPN

# 5. Media Server (ID 102)

I had PLEX running on my Raspberry Pi. For the server I tried Turnkey Mediaserver with Emby, however it was painfully slow on our Samsung SmartTV, so had to switch back to PLEX. I installed it from apt repository on Debian container. Obviously mount the shared folders and open them from inside the app.

# 5. Turnkey Nextcloud

Ended up not mounting any host folders and leaving it isolated as is. Use Turnkey `confconsole` to get letsencrypt certificate for the domain.

# 5a. Seafile option
Use centos and script, do "yum install which" to prevent python check from failing before executing script.

# 6. Monitoring

I am using TIG (telegraf-influxdb-grafana) stack for monitoring at the host level. Grafana and influxdb run as LXC, while telegraf runs at host level to get access to system-level parameters.

List of monitoring related services with notes is below.

#### 6a. InfluxDB (ID 104)

Influxdb runs inside unpriviliged Alpine LXC in a tiny container (1Gb space, 256Mb RAM). Install using apk and add to [startup scripts](https://www.cyberciti.biz/faq/how-to-enable-and-start-services-on-alpine-linux/). I had an issue with service crashing, as the `/etc/network/interfaces` was missing the `lo loopback interface`. Don't forget to enable the HTTP binding in `/etc/influxdb/influxdb.conf`, otherwise I left configuration untouched. Also add to Proxmox level backup and auto-startup.

#### 6b. Grafana (ID 105)

You can run a compiled binary on Alpine if you [install glibc](https://github.com/sgerrand/alpine-pkg-glibc). However then you have to create a service to run at boot time, add user, so on. So I decided to use familiar debian LXC instead. Installed using apt-get. Don't forget to enable systemctl service to start on boot.

#### 6c. Telegraf

Install directly on the host, move configuration file to directory on ZFS pool to enable backup and symlink to `/etc/telegraf/`. Point influxdb in configuration file to IP from 7a.

#### 6d. Additional monitoring utilities

Some additional utilities to get parameters not available directly in telegraf:
- `hddtemp` for drive temperature monitoring
- [Proxmox external metrics](https://pve.proxmox.com/wiki/External_Metric_Server), influxdb has to have UDP enabled.
- `nvidia-smi` is available oob, see instructions below on GPU part

# 7. Web Server (ID 107)

Running nodejs web server in Alpine LXC. Steps to set up are the following. Update apk and install openssh and sudo. Add sudo user (visudo) for ssh access. Install node `apk add nodejs nodejs-npm` and pm2 by `npm install -g pm2`. Create a non-sudo `www` user. Mount your web app folder from proxmox host zpool and chmod to 777 to allow read/write to everyone. Generate pm2 startup scripts for www user. Launch all nodejs apps and save them with pm2 to persist. pm2 will also be responsible for restarting apps.

For SSL setup install certbot. NodeJS app needs to be refactored to serve a static folder, e.g. `public` to allow certbot to create challenges. After this the certificate can be generated by:
```
sudo certbot certonly --webroot -w /var/www/e2pe/public -d e2.pe --config-dir /var/www/certificates
```
It will create certificates in the host folder, so they can be shared with reverse proxy.
Then create a root cronjob for daily renew attempts `sudo crontab -e` and add `certbot renew --config-dir /var/www/certificates`.

This [tutorial](https://www.sitepoint.com/how-to-use-ssltls-with-node-js/) has some other details for HSTS and DH options. Create one stronger DH cert shared between nginx reverse proxy and every subdomain or app.

One such LXC can take care of multiple apps - just need to assign different ports, so that nginx reverse proxy can redirect correspondingly.

# 7. Python / Deep Learning (ID 200)

[This guide](https://medium.com/@MARatsimbazafy/journey-to-deep-learning-nvidia-gpu-passthrough-to-lxc-container-97d0bc474957) has a lot of good information, but needs few adjustments.

First need to install on host. Use [official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Get proper headers first. I ended up installing using `apt-get install -t stretch-backports nvidia-cuda-toolkit` from `deb http://httpredir.debian.org/debian stretch-backports main contrib non-free` as per [this topic](https://unix.stackexchange.com/questions/218163/how-to-install-cuda-toolkit-7-8-9-on-debian-8-jessie-or-9-stretch). Make sure to update `stretch` to correct debian/proxmox version. Verity installation by `cat /proc/driver/nvidia/version` and `nvcc -V`.

Enable start at boot as per [Nvidia guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-verifications). Create a file in `/etc/init.d` and add to startup scripts.

Share with LXC as per the guide, linked in the beginning. Then install `nvidia-cuda-toolkit` similar as on host. Verify that `nvidia-smi` and `nvcc -V` give the same versions as on the host. Download and install the cuDNN deb package. Look at archived section in case you need to match CUDA version exactly.

Will use Tensorflow as computing backend. Tensorflow [recommends (or requires?)](https://www.tensorflow.org/install/gpu) GPUs "compute capability" of at least 3.5. Unfortunately my refurbished GTX650 has 3.0, but some posts on stack overflow indicate that it is still working with 3.0. I didn't bother with compiling tensorflow and just used `pyenv` to install miniconda. Miniconda in turn provides a pre-compiled TF with GPU support by `conda install -c anaconda tensorflow-gpu`, which worked oob.

Keras MNIST CNN example output:
```
2018-12-24 20:51:10.840424: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: GeForce GTX 650 major: 3 minor: 0 memoryClockRate(GHz): 1.0585
pciBusID: 0000:01:00.0
totalMemory: 978.12MiB freeMemory: 954.25MiB
2018-12-24 20:51:10.840468: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2018-12-24 20:51:11.156234: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-12-24 20:51:11.156289: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2018-12-24 20:51:11.156298: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2018-12-24 20:51:11.156526: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 683 MB memory) -> physical GPU (device: 0, name: GeForce GTX 650, pci bus id: 0000:01:00.0, compute capability: 3.0)
60000/60000 [==============================] - 34s 565us/step - loss: 0.2767 - acc: 0.9152 - val_loss: 0.0588 - val_acc: 0.9813
Epoch 2/12
60000/60000 [==============================] - 31s 519us/step - loss: 0.0902 - acc: 0.9737 - val_loss: 0.0472 - val_acc: 0.9842
Test loss: 0.024805454400579036
Test accuracy: 0.9917
```

According to Grafana it took about 6min of GPUs time with temperature raising up to 53 degC. The average time of epoch is 31s compated to an average of 142s on my mid-2015 13" MacBook Pro with ~330% CPU load and associated noise and heat, so not bad for a $50 card.

# 8. Backup

Use homelab machine as borg backup server for all computers inside the network, as well as for itself. Then I use rclone to sync the deduplicated borg repository with B2. How to [daemonize rclone](https://forum.rclone.org/t/rclone-daemonized/648/8).