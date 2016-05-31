---
layout: post
type: post
title: 使用本地DNS和SOCKS5代理服务器改善访问受限网络服务
tags: network, local dns, socks5
description: 几年前买的RaspberryPi A+一直在闲置，每小时不到1W的功耗让它可以胜任很多有用的小服务。继在上面mount了一个基于evdev的Xbox360 Controller Accepter, 前段时间我在Pi上部署了另外几项服务，用来提升访问中国以外地区网络服务的体验。我明显的留意到，这个新的方案带来的改善明显超过了过去几年我使用的若干种方案。
---

几年前买的RaspberryPi A+一直在闲置，每小时不到1W的功耗让它可以胜任很多有用的小服务。继在上面mount了一个基于evdev的Xbox360 Controller Accepter, 前段时间我在Pi上部署了另外几项服务，用来提升访问中国以外地区网络服务的体验。我留意到，这个新的方案带来的改善明显超过了过去几年我使用的若干种方案。

比起其它方案，这个方案的好处主要有这样几点:

1. 不需要开关配置（比如像VPN那样，用户需要在需要访问受限服务时主动打开VPN)
1. 支持更多种类的设备，比如Sony PS4
1. 非常稳定（中断的连接只影响当前请求，新发生的请求将自动使用重建的连接，不像VPN）
1. 基于IP段的域名解析（而非基于域名查询结果的），不需要使用一个人工维护的域名列表。从IP段中得到服务的位置信息，然后为所有可能的受限服务分配代理。进一步提高可靠性。
1. 优化了的域名解析。境内和境外的域名使用不同的DNS。
1. 代理服务器使用基于事件的异步IO，保证了网络的稳定性和可靠响应。
1. 加密的连接不易被审查程序察觉，没有显著的外部特征。这一点保证了这个方案的生命力。

### DNS欺骗和连接重置

当境内用户访问境外的网络服务时，这是两种经常发生的网络状况。ISP的机房中的DNS中的审查程序会在一个连接还没有建立之前，在域名解析阶段误导用户的连接请求。错误的域名解析结果不仅使得连接失败，有时候大量来自用户的请求也可以被集中重定向用来实施分布式拒绝服务攻击(DDoS)。对于那些躲过DNS欺骗的连接，审查程序会对经过的数据进行探测（Traffic Analysis），然后选择性的将数据包丢弃。比如通过对请求头的分析，审查程序发现加过密的连接在尝试访问禁用的服务（流行的加密传输协议加密后的数据通常带有非常明显的外部特征），在不能够破解加密内容的情况下，审查程序干脆就将数据包丢弃，阻止连接传输数据（这会影响像通过ssh -D这样的方案）。

### 流量分析(Traffic Analysis)

很显然，未经加密的明文协议，比如HTTP，是没有办法做到安全和匿名的。所以审查程序真正的挑战是那些加密的传输协议。简单的流量分析可以从阅读数据包的请求头开始，这非常容易。挑战是加密传输协议，比如SSL(TLS)，使用混合加密方案。在连接建立之初的验证阶段，连接双方使用基于公钥私钥的非对称加密，这个过程中会为当前回话沟通一个临时的密码，用来为接下来的传输进行对称加密。虽然这阻止了审查程序查看传输内容，但是这种加密方式加密后的内容会在传输数据中留下一些外部特征。通过使用一些机器学习的手段（kNN, Bayesian Inference)，审查程序可以从传输内容中大概的总结出这些规律。这些信息有时候会帮助它做出决定，是否阻止当前连接。

### VPN

VPN是使用的非常广泛的一种方案。VPN工作在IP层，对IP数据包中的数据进行加密，但是请求头是未经加密的（改进的版本中，请求头也被加密了）。通常，用户需要在终端设备上开关VPN设置，以单独访问境内和境外的内容。同时消费者VPN客户端的稳定性都不太好，缺少比如重建中断连接这样的功能。另外，使用VPN时，默认的域名解析将使用远程的服务器解析，网络延迟非常的大。

### ssh -D

加密的本地端口转发曾经也是一个非常不错的方案。优点包括可靠的加密以及极其简易的使用方式。但是经由RSA混合加密的数据具有非常明显的外部特征。这一点使得它在进化了的审查程序面前的稳定性打了折扣。同时，ssh -D作为一种简单的方案，它缺乏自建中断连接，多连接这些特性，从而也并不是一个高效和可靠的方案。另外，家庭中的其它设备难以共享这个本地转发的端口。

### 各个击破

![domain_name_rule_based_selectively_encrypted_traffic.jpeg](/images/2016-05-30-a-local-dns-and-proxy-on-raspberrypi/domain_name_rule_based_selectively_encrypted_traffic.png)

如图所示，当一个连接请求发生时:

1. 终端设备访问本地DNS进行域名解析
1. 本地DNS通过解读IP段的规则，分别访问境外的DNS用来解析境外域名，而访问境内DNS解析境内域名
1. 终端设备成功的获得未经污染的域名解析结果后，访问本地提供代理配置文件的HTTP服务，来决定是否使用SOCKS5代理建立当前连接
1. 代理配置文件服务使用一个动态生成的列表，来决定是否为当前连接使用SOCKS5代理。这个列表是经由分析IP段生成的(https://github.com/Leask/Flora_Pac)。
1. 如果代理配置为当前连接指示了SOCKS5代理，终端设备将访问本地的SOCKS5代理服务器，尝试建立连接
1. 当本地SOCKS5代理服务器收到连接请求，它会访问远程的通常部署在境外的SOCKS5服务器，尝试建立连接。这是一个关键的部分。在本地SOCKS5代理服务器和远程SOCKS5服务器建立连接指出，双方会就一些连接细节进行协商。包括登陆密码和将采用的加密方式等。注意，这两者之间的连接是自加密的SOCKS5协议内容，加密方式有好几种。加密方式的选择和设计试图规避审查程序对依赖于加密数据外部特征的包探测。同时通过分析该项目的源代码（https://github.com/Long-live-shadowsocks/shadowsocks/blob/master/shadowsocks/tcprelay.py）， 软件的实现使用了基于事件的异步IO，并且为每一个代理请求分配了单独的连接，这个方案的稳定性和效率将在实际使用中得到充分的体现。

### 部署

使用一台RaspberryPi (系统: Linux raspberrypi):

- Pi通过网线连接在路由器（这样可以杜绝丢包产生的延迟，因为Pi上提供的网络服务非常多）
- 在路由器的设置页面为Pi分配固定的IP（因为服务配置的过程中需要多次使用Pi的地址，你不希望这个地址变动)

以下是各个服务的部署过程:

#### Local DNS (ChinaDNS)

```
# Check out the source code of ChinaDNS project
$ git clone https://github.com/shadowsocks/ChinaDNS.git
# Build
$ cd ChinaDNS
$ ./configure && make
# Run the DNS on the default port
$ sudo src/chinadns -m -c chnroute.txt
# If the service succeeds, you can see with lsof:
$ sudo lsof -i :53
COMMAND  PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
chinadns 439 root    3u  IPv4   8622      0t0  UDP *:domain
# You can also verify it by using this DNS for a name query:
# On a RaspberryPi Linux distro, you need to run apt-get install dnsutils
$ dig @127.0.0.1 -x google.com
# Edit /etc/rc.local and add the following to launch service at startup:
/ABSOLUTE_PATH_TO_CHINADNS/src/chinadns -m -c /ABSOLUTE_PATH_TO_CHINADNS/chnroute.txt >> /dev/null &
```

Take a note of the address.

#### Local HTTP Proxy Configuration

```
# Checkout the source code of flora_pac project
$ git clone https://github.com/Leask/Flora_Pac.git
# Run the HTTP server to serve the dynamically generated PAC
$ ./flora_pac -p 8970
# Edit /etc/rc.local and add the following to launch service at startup:
export PROXY_SOCKS='SOCKS5 192.168.1.111:1080; SOCKS 192.168.1.111:1080; DIRECT'
/ABSOLUTE_PATH_TO_FLORA_PAC/flora_pac -x '$PROXY_SOCKS' -p 8970 >> /dev/null &
```

Take a note of the address and the port.

#### Remote SOCKS5 Server

The **remote socks5 server** is what saves the world. It must be a host that directly has access to the unrestricted service (An ideal choice would be a VPS, or an IaaS basic configuration that usually costs 5~15 USD per month. An important choice during configuring such a service is to decide the region. A popular choice is to locate your host in Singapore or Tokyo if you are living in Chinese mainland).

The [shadowsocks](https://pypi.python.org/pypi/shadowsocks) project is a Python implementation of SOCKS5 protocol. The author took some specific considerations and made it particularly effective against censorship based on package sniffing.

Its highlights include:

- A Python code base that is very easy to work with and adapt
- Event-based Asyncio that made it reliable under high concurrency
- It creates connection for every request
- It supports several strategically selected encryptions that aims to encrypt the traffic without giving away too much fingerprints. Beyond this, it can randomly pick one of these encryptions during a session

```
# Install the shadowsocks with pip
$ pip install shadowsocks
# After a successful installation, you have two main commands, the ssserver and sslocal
# Create a configuration file
$ mkdir /etc/shadowsocks
$ cd /etc/shadowsocks
$ vim config.json
# For example, here is how my server config.json looks like:
{
"server":"THE_REMOTE_SERVER_IP_ADDRESS",
"server_port":8388,
"local_address": "0.0.0.0",
"local_port":1080,
"password":"PASSWORD_AT_YOUR_WILL",
"timeout":300,
"method":"aes-256-cfb",
"fast_open": true,
"workers": 5
}
# Launch the service
$ ssserver -c config.json
# If your system uses upstart, which is my case, the following in a file /etc/init/shadowsocks.conf will launch the service at startup:
exec ssserver -c /etc/shadowsocks/config.json > /dev/null &
```

Take a note of the server address and server port.

#### Local SOCKS5

If you still have that picture of the architecture in your mind, the **local socks5** is the service that sits **between** the end-device and the remote server. It is itself simultaneously a client (in terms of the remote server) and a server (in terms of your end-device such as mobile phone). It is a **relay** in telecommunication terms.

Here below is how we launch this service:

```
# Install the shadowsocks with pip
$ pip install shadowsocks
# After a successful installation, you have two main commands, the ssserver and sslocal
# Before we use sslocal to launch the local end of connection, we create a configuration file:
$ mkdir /etc/shadowsocks
$ cd /etc/shadowsocks
$ vim config.json
# For example, this is what my config.json looks like:
{
  "server": "THE_REMOTE_SERVER_IP_ADDRESS",
  "server_port": 8388,
  "local_address": "0.0.0.0",
  "local_port": 1080,
  "password": "PASSWORD_AT_YOUR_WILL",
  "timeout": 300,
  "method": "aes-256-cfb"
}
# Now launch the local end of connection
$ sslocal -c config.json
# Edit /etc/rc.local and add the following line to launch service at startup:
sslocal -c /etc/shadowsocks/config.json >> /dev/null &
```

Take a note of the address and the port in the config.json.

#### Clients

If all services above were successfully setup, you are almost there. Now you just have to setup your device:

1. Configure your device to use a different primary DNS. Locate the network settings of your device and point the primary DNS to the address of the host where you setup the DNS. In my case, it is the RaspberryPi hardwired on the router.
1. Configure your decide to use a **global automatic proxy**. In my case with an iMac and Apple iOS devices, locate the network settings, and update the proxies to use an Automatic Proxy Configuration, and fill in with the service address you used in the step of launching the **Local HTTP Proxy Configuration**.

At this point, your Internet traffic will:

1. Resolve names with a DNS local to your intranet.
1. Allocates a proxy for all foreign names.
1. The proxy allocated is an encrypted SOCKS5 connection, very difficult for the censorship to sniff.
1. Use an un-proxied connection for all domestic services.

### 声明

本文所述所有内容只作为学习用途。作者不对任何采用文中内容所导致的后果负责。

