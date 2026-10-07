const {chromium}=require(process.env.PLAYWRIGHT_MODULE ?? '/opt/codex/runtimes/cua/lib/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-gpu']});
 for (const width of [1440,390]) {
  const page=await browser.newPage({viewport:{width,height:900}}); let calls=[];
  await page.route('**/v1/images',route=>route.fulfill({json:{images:[]}}));
  await page.route('**/v1/images/generations',async route=>{ calls.push(route.request().postDataJSON()); await route.fulfill({json:{image:{id:'native',mode:'Generate',prompt:'Tree',aspect:'square',seeds:[7],meta:'Qwen-Image-2.1',urls:['/fixture.png']}}}); });
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j3ioAAAAASUVORK5CYII=','base64');
  await page.route('**/fixture.png',r=>r.fulfill({contentType:'image/png',body:png}));
  await page.goto('http://127.0.0.1:15238/image');
  await page.getByPlaceholder('Describe the image you want…').fill('Tree');
  await page.getByRole('button',{name:'1',exact:true}).click();
  await page.getByPlaceholder('Random').fill('7');
  await page.getByRole('button',{name:'Generate 1 image',exact:true}).click();
  await page.getByRole('img',{name:'Tree',exact:true}).waitFor();
  assert.equal(calls.length,1); assert.equal(calls[0].seed,7);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:`/tmp/qwen-image-${width}.png`,fullPage:true});
  await page.goto('http://127.0.0.1:15238/image/edit');
  const strength=page.getByRole('slider',{name:'Edit strength',exact:true});
  assert.equal(await strength.isVisible(),true); assert.equal(await strength.isDisabled(),true);
  assert.equal(await strength.getAttribute('aria-describedby'),'edit-strength-status');
  assert.match(await page.locator('#edit-strength-status').innerText(),/not supported/);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:`/tmp/qwen-edit-strength-${width}.png`,fullPage:true});
  await page.locator('input[type=file]').setInputFiles({name:'fixture.png',mimeType:'image/png',buffer:png});
  await page.getByPlaceholder('e.g. Replace the background with a sunlit studio, keep the subject unchanged').fill('Blue background');
  await page.route('**/v1/images/edits',async route=>{
   const body=route.request().postDataJSON();assert.match(body.image,/^data:image\/png;base64,/);assert.equal('strength' in body,false);
   await route.fulfill({status:503,json:{detail:'Fixture GPU failure'}});
  });
  await page.getByRole('button',{name:'Apply edit',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'Fixture GPU failure'}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Apply edit',exact:true}).isEnabled(),true);
  await page.close();
 }
 await browser.close(); console.log('Desktop/mobile generation, PNG gallery, source upload, backend failure and retry state passed.');
})();
