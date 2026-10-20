const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function fixture() {
    const nodes = {};
    const document = {getElementById(id) {return nodes[id] ||= {textContent:'',setAttribute(k,v){this[k]=v;}};}};
    const context = {document, safeSetText(id,text){document.getElementById(id).textContent=text;},fmtNum:String};
    vm.createContext(context);
    vm.runInContext(fs.readFileSync('web/js/predictor.js','utf8'),context);
    return {nodes, render:context.renderHistoricalReplay};
}
const paper = {paper_id:'W123',publication_year:2020,cutoff_year:2022,split:'test',predicted_label:'Medium',backtest:{has_ground_truth:true,actual_label:'Low',actual_citations_window:0,is_correct_class:false,window_start:2023,window_end:2025}};
test('zero-citation outcome is valid; reveal reports failure and next paper resets it',()=>{
    const {nodes,render}=fixture(); render(paper);
    assert.equal(nodes.replayActual.hidden,true);
    assert.equal(nodes.revealOutcomeButton.hidden,false);
    nodes.revealOutcomeButton.onclick();
    assert.equal(nodes.replayActual.hidden,false);
    assert.match(nodes.replayVerdict.textContent,/Did not match/);
    render(paper);
    assert.equal(nodes.replayActual.hidden,true);
    assert.equal(nodes.revealOutcomeButton.disabled,false);
});
test('unverified outcome never offers reveal or a correctness verdict',()=>{
    const {nodes,render}=fixture(); render({...paper,split:'external',backtest:{has_ground_truth:false,warning:'Window incomplete'}});
    assert.equal(nodes.revealOutcomeButton.hidden,true);
    assert.equal(nodes.replayVerdict.textContent,'');
    assert.match(nodes.replayUnavailable.textContent,/Window incomplete/);
});
test('training examples explicitly disclose membership',()=>{
    const {nodes,render}=fixture(); render({...paper,split:'train'});
    assert.match(nodes.replayMembership.textContent,/cannot demonstrate unseen-paper accuracy/);
});
