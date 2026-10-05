const {chromium}=require('/opt/codex/runtimes/cua/lib/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 for (const width of [1440,390]) {
  const page=await browser.newPage({viewport:{width,height:900}}); let calls=[];
  await page.route('**/v1/images',route=>route.fulfill({json:{images:[]}}));
  await page.route('**/v1/images/generations',async route=>{ calls.push(route.request().postDataJSON()); await route.fulfill({json:{image:{id:'native',mode:'Generate',prompt:'Tree',aspect:'square',seeds:[],meta:'FLUX 3 Image',urls:['/fixture.png']}}}); });
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j3ioAAAAASUVORK5CYII=','base64');
  await page.route('**/fixture.png',r=>r.fulfill({contentType:'image/png',body:png}));
  await page.goto('http://127.0.0.1:15243/image');
  await page.getByPlaceholder('Describe the image you want…').fill('Tree');
  await page.getByRole('button',{name:'1',exact:true}).click();
  await page.getByLabel('Model',{exact:true}).selectOption('flux-3-image');
  assert.equal(await page.getByPlaceholder('Unavailable').isDisabled(),true);
  await page.getByRole('button',{name:'Generate 1 image',exact:true}).click();
  await page.getByRole('img',{name:'Tree',exact:true}).waitFor();
  assert.equal(calls.length,1); assert.equal(calls[0].seed,null); assert.equal(calls[0].model,'flux-3-image');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:`/tmp/flux-image-${width}.png`,fullPage:true});
  await page.goto('http://127.0.0.1:15243/image/edit');
  await page.getByLabel('Model',{exact:true}).selectOption('flux-3-image');
  const strength=page.getByRole('slider',{name:'Edit strength'});
  assert.equal(await strength.isVisible(),true); assert.equal(await strength.isDisabled(),true);
  assert.equal(await strength.getAttribute('aria-describedby'),'edit-strength-status');
  await page.getByText('Edit strength is not supported by this model.',{exact:true}).waitFor();
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
