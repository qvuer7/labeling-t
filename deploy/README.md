# Deploy: hosted Label Studio on a DigitalOcean droplet

A self-provisioning Label Studio host. You never run setup commands on the
server — the droplet's **cloud-init** (`cloud-init.yaml`) writes the stack and
starts it on first boot. Frames load from S3 presigned URLs, so the cloud stack
is just **Label Studio + Caddy** (auto-HTTPS, no DNS needed via `nip.io`).

## Launch — pick one

### A) DigitalOcean web UI (one click)
1. Create Droplet → **Marketplace image: "Docker"** (Docker preinstalled).
2. Size: 2 vCPU / 2–4 GB is plenty. Region: **fra1** (same as your Spaces bucket).
3. Add your SSH key.
4. **Advanced options → Add Initial Scripts (user data)** → paste the contents of
   `cloud-init.yaml`.
5. Create. Wait ~2–3 min, then SSH once to read the URL + credentials:
   ```
   ssh -i ~/.ssh/docean_droplet root@<ip> cat /root/LABELSTUDIO.txt
   ```

### B) doctl / API (fully scripted)
```bash
doctl compute droplet create labelstudio \
  --image docker-20-04 --size s-2vcpu-4gb --region fra1 \
  --ssh-keys <your-key-fingerprint> \
  --user-data-file deploy/cloud-init.yaml --wait
# then:
doctl compute ssh labelstudio --ssh-command "cat /root/LABELSTUDIO.txt"
```

### C) Manual (if you prefer)
```bash
scp -r deploy/{docker-compose.yml,Caddyfile} root@<ip>:/opt/labelstudio/
ssh root@<ip>
cd /opt/labelstudio && cp <fill> .env && docker compose up -d   # see .env.example
```

## After it's up (one command, from your laptop)

Label Studio 1.23+ disables SDK (legacy) tokens by default, so enable them once
and grab the token (works against any LS URL):
```bash
LS_URL=https://<ip-with-dashes>.nip.io \
LS_EMAIL=admin@labeling-t.local LS_PASSWORD=<from LABELSTUDIO.txt> \
  uv run python scripts/ls_setup.py        # prints the API token
```

Then point the backend at the hosted instance — no code change, just the URL:
```bash
labeling-t import-ls --url https://<domain> --api-key <token> \
    --project ipbl-basketball --categories player,ball,referee \
    --image-base-url <presigned S3 base>   # see note below
```

## Note: images

The hosted LS loads frames from **S3 presigned URLs**, not a local image server.
The `import-ls` presigned-image path (Phase 3) is the small remaining code bit;
until then, frames must be referenced by a URL the browser can reach.

## Costs / notes

- Droplet: a 2 vCPU / 4 GB box is a few $/mo. Annotations persist in the
  `ls-data` Docker volume (snapshot the droplet or back up to S3 for safety).
- `nip.io` + Let's Encrypt has rate limits; for a permanent setup point a real
  domain at the droplet and set `LS_DOMAIN` to it.
- Lock it down: the droplet exposes 80/443 publicly — fine behind LS login, but
  consider a firewall allowing only your IP if it's just you.
