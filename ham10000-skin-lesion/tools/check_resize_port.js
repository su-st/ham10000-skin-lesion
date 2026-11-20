// 由 check_resize_port.py 呼叫
const fs = require('fs'), path = require('path');
const L = require(path.join(__dirname, '..', 'prototype_web', 'lesion.js'));
const D = process.argv[2], cases = JSON.parse(fs.readFileSync(path.join(D, 'cases.json')));
let bad = 0;
for (const c of cases) {
  const src = new Uint8Array(fs.readFileSync(path.join(D, `in${c.i}.bin`)));
  const exp = new Uint8Array(fs.readFileSync(path.join(D, `out${c.i}_${c.ow}.bin`)));
  const got = L.resize(src, c.w, c.h, c.ch, c.ow, c.oh);
  let diff = got.length === exp.length ? 0 : exp.length;
  for (let k = 0; k < exp.length; k++) if (exp[k] !== got[k]) diff++;
  if (diff) { bad++; console.log('不一致', c, diff); }
}
console.log(`${cases.length} 組測試，${bad ? bad + ' 組不一致' : '全部與 PIL 逐像素相同'}`);
process.exit(bad ? 1 : 0);
