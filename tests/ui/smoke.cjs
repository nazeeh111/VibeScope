/* Run against an already started VibeScope instance. No mocked measurements. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const url = process.env.VIBESCOPE_URL || 'http://127.0.0.1:8765';
(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROME_EXECUTABLE || undefined});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1080},baseURL:url});
    const errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(url);
    await page.locator('#demoButton').click();
    await page.waitForFunction(()=>!document.querySelector('#analyzeButton').disabled,null,{timeout:180000});
    assert.match(await page.locator('#sourceKind').innerText(),/SYNTHETIC/);
    assert.ok(await page.locator('#analysisForm').evaluate(form=>form.checkValidity()),'Demo controls must pass native form validation');
    await page.locator('#analyzeButton').click();
    await page.waitForFunction(()=>/Measurement (complete|failed)/.test(document.querySelector('#jobStatus').textContent),null,{timeout:180000});
    assert.ok(await page.locator('#errorBanner').isHidden(),await page.locator('#errorBanner').textContent());
    const frequency=Number(await page.locator('#frequencyValue').innerText());
    assert.ok(Math.abs(frequency-7.3)<.1,`Recovered demo frequency ${frequency}`);
    await page.locator('#originalVideo').evaluate(video=>video.readyState>=2?undefined:new Promise(resolve=>video.addEventListener('loadeddata',resolve,{once:true})));
    await page.locator('#enhancedVideo').evaluate(video=>video.readyState>=2?undefined:new Promise(resolve=>video.addEventListener('loadeddata',resolve,{once:true})));
    await page.locator('#playButton').click();
    await page.waitForTimeout(600);
    let media=await page.evaluate(()=>[document.querySelector('#originalVideo').currentTime,document.querySelector('#enhancedVideo').currentTime,document.querySelector('#originalVideo').paused]);
    assert.ok(media[0]>.1&&!media[2]&&Math.abs(media[0]-media[1])<.15,`Synchronized playback: ${media}`);
    await page.locator('#playButton').click();
    await page.locator('#timeline').fill('1.2');
    await page.locator('#timeline').dispatchEvent('input');
    await page.waitForTimeout(150);
    media=await page.evaluate(()=>[document.querySelector('#originalVideo').currentTime,document.querySelector('#enhancedVideo').currentTime]);
    assert.ok(media.every(t=>Math.abs(t-1.2)<.1),'Timeline seeks both videos');
    for(const id of ['exportCsv','exportPlot','exportReport']){
      const href=await page.locator('#'+id).getAttribute('href');
      const response=await page.request.get(href);
      assert.ok(response.ok()&&(await response.body()).length>100,`${id} returns a real artifact`);
    }
    const baselinePath = process.env.VIBESCOPE_BASELINE_JSON || path.resolve(__dirname,'../../experiments/controlled/synthetic_reference.json');
    if (fs.existsSync(baselinePath)) {
      await page.locator('#baselineConfig').evaluate(details=>{details.open=true;});
      await page.locator('#baselineFile').setInputFiles(baselinePath);
      await page.waitForFunction(()=>/Reference (loaded|import failed)/.test(document.querySelector('#jobStatus').textContent),null,{timeout:30000});
      assert.ok(await page.locator('#errorBanner').isHidden(),await page.locator('#errorBanner').textContent());
      assert.equal(await page.locator('#baselineConfirmed').isChecked(),false);
      await page.locator('#analyzeButton').click();
      assert.match(await page.locator('#errorBanner').innerText(),/Confirm the same acquisition setup/);
      await page.locator('#baselineConfirmed').check();
      await page.locator('#analyzeButton').click();
      await page.waitForFunction(()=>/Measurement (complete|failed)/.test(document.querySelector('#jobStatus').textContent),null,{timeout:180000});
      assert.ok(await page.locator('#errorBanner').isHidden(),await page.locator('#errorBanner').textContent());
      const response=await page.request.get(await page.locator('#exportReport').getAttribute('href'));
      const report=await response.json();
      assert.equal(report.metrics.anomaly.diagnostic,false);
      assert.equal(report.metrics.anomaly.acquisition_match,'user_confirmed');
      assert.match(await page.locator('#anomalyResult').innerText(),/Reference distance/);
      await page.locator('#axis').selectOption('y');
      assert.equal(await page.locator('#baselineConfirmed').isChecked(),false,'Changing acquisition invalidates reference confirmation');
      await page.locator('#axis').selectOption('x');
      await page.locator('#clearBaseline').click();
      assert.match(await page.locator('#baselineSummary').innerText(),/No reference loaded/);
      console.log('PASS: real reference import, explicit acquisition confirmation, non-diagnostic scoring and context invalidation.');
    }
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile page must not overflow horizontally');
    await page.setViewportSize({width:1440,height:1080});
    await page.locator('#calibration').fill('0.1');
    await page.locator('#ringdown').check();
    await page.locator('#demoButton').click();
    await page.waitForFunction(()=>!document.querySelector('#analyzeButton').disabled,null,{timeout:180000});
    assert.equal(await page.locator('#calibration').inputValue(),'','Calibration cannot carry across sources');
    assert.equal(await page.locator('#ringdown').isChecked(),false,'Ringdown declaration cannot carry across sources');
    await page.locator('#lowHz').fill('12');
    await page.locator('#highHz').fill('8');
    await page.locator('#analyzeButton').click();
    assert.match(await page.locator('#errorBanner').innerText(),/Choose 0 < low < high/);
    assert.deepEqual(errors,[]);
    console.log('PASS: real demo measurement, native form validity, synchronized playback, seek, exports, mobile layout, source-context reset and invalid-band feedback.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
