import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
const browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_PATH || undefined,headless:true,args:['--no-sandbox']});
for(const viewport of [{width:1440,height:1000},{width:390,height:844}]) {
const page=await browser.newPage({viewport}); let payload;
await page.route('**/v1/videos',r=>r.fulfill({json:{jobs:[]}}));
await page.route('**/v1/videos/generations',async r=>{payload=r.request().postDataJSON();await r.fulfill({status:503,json:{detail:'Worker unavailable'}});});
await page.goto(`${process.env.BASE_URL || 'http://127.0.0.1:5173'}/video`);
await page.getByLabel('Prompt',{exact:true}).fill('A river');
assert.equal(await page.getByLabel('Model',{exact:true}).inputValue(),'h3-fl2va');
assert(await page.getByLabel('Negative prompt').isDisabled());
await page.getByRole('button',{name:'Queue video'}).click();
await page.getByText('Video generation is unavailable.',{exact:false}).waitFor();
assert.equal(payload.model,'h3-fl2va');assert.equal(payload.resolution,'768p');assert.equal(payload.fps,24);
assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
if (process.env.SCREENSHOT_DIR) await page.screenshot({path:`${process.env.SCREENSHOT_DIR}/h3-${viewport.width}.png`,fullPage:true});
await page.close();
}
await browser.close();console.log('H3 desktop/mobile request and failure UI passed');
})();
