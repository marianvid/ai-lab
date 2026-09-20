export async function readBase64(file, maxBytes, label) {
  if (!file) throw new Error(`Select a ${label}`);
  if (file.size > maxBytes) throw new Error(`${label} exceeds 25 MiB`);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const chunks = [];
  for (let offset = 0; offset < bytes.length; offset += 32768) {
    chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 32768)));
  }
  return btoa(chunks.join(''));
}
