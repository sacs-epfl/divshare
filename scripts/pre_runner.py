import json

with open("/selfip.txt") as sip:
	self_ip = sip.read().splitlines()[0]

with open("/slowips.txt") as ips:
	all_ips = ips.read().splitlines()

with open("/fastips.txt") as ips:
	all_ips += ips.read().splitlines()

with open("/identity.txt", "w") as id:
	identity = all_ips.index(self_ip)
	id.write(str(all_ips.index(self_ip)))

tojson = dict()
for ind, elem in enumerate(all_ips):
	tojson[ind] = elem

with open("/all_ips.json", "w") as allips:
	json.dump(tojson, allips) 
	allips.write("\n")

with open("/num_workers.txt", "w") as nw:
	nw.write(str(len(all_ips)))
