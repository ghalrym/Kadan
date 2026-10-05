const {chromium}=require('/opt/codex/runtimes/cua/lib/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{ const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
try { for(const width of [1440,390]) {
 const page=await browser.newPage({viewport:{width,height:1000}});let selected=null,puts=0,fail=false;
 const models=['qwen-image-2.1','flux-klein-4b'].map(id=>({id,kind:'image',repo_id:id,display_name:id==='flux-klein-4b'?'FLUX.2 klein 4B':'Qwen-Image-2.1',revision:'fixture',license:'apache-2.0',estimated_bytes:16e9,inference_available:true,status:'complete',downloaded_bytes:16e9,total_bytes:16e9,error:null,context_limit:null,architecture_context_limit:null,license_notice:null,license_url:null}));
 await page.route('**/model-lifecycle',r=>r.fulfill({json:{state:'unloaded',model_id:null,error:null}}));
 await page.route('**/v1/models**',async r=>{if(r.request().method()==='PUT'){puts++;if(fail)return r.fulfill({status:503,json:{detail:'Storage unavailable'}}); selected=r.request().postDataJSON().model_id;}await r.fulfill({json:{models,selected_model_id:null,selected_image_model_id:selected}})});
 await page.goto('http://127.0.0.1:15246/settings');
 await page.locator('#model-Image').click();await page.getByRole('button',{name:/^FLUX\.2 klein 4B/}).click();
 await page.waitForFunction(()=>document.querySelector('#model-Image')?.textContent.includes('FLUX.2 klein 4B'));assert.equal(selected,'flux-klein-4b');assert.equal(puts,1);
 await page.reload();await page.locator('#model-Image').filter({hasText:'FLUX.2 klein 4B'}).waitFor();
 fail=true;await page.locator('#model-Image').click();await page.getByRole('button',{name:/^Qwen-Image-2.1/}).click();await page.getByText('Storage unavailable',{exact:true}).waitFor();assert.equal(selected,'flux-klein-4b');assert.equal(await page.locator('#model-Image').innerText(),'FLUX.2 klein 4B');
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 await page.screenshot({path:`/tmp/klein-selection-${width}.png`,fullPage:true});await page.close();
 }console.log('Desktop/mobile image selection, reload persistence, failed-selection preservation passed');
}finally{await browser.close()}})();
