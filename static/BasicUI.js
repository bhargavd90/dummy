var nodes;
var edges;
var network;
var cluster_dict;
var docs_dict;
var text_dict;
var related_events;
var title_dict;
var summary_dict;
var place_dict;
var person_dict;
var date_dict;
var possible_content_depth;
var news_path;
var summary;

var modal_news = document.getElementById('modalNews');
var modal_news_content = document.getElementById('modalNewsContent');
var modal_news_header = document.getElementById('modalNewsHeader');
var modal_settings = document.getElementById('modalSettings');
var modal_settings_content = document.getElementById('modalSettingsContent');
var modal_settings_header = document.getElementById('modalSettingsHeader');
var close_news = document.getElementsByClassName("close_news")[0];
var close_settings = document.getElementsByClassName("close_settings")[0];

var slider1 = document.getElementById("myRange1");
var output1 = document.getElementById("demo1");
var slider2 = document.getElementById("myRange2");
var output2 = document.getElementById("demo2");
output1.innerHTML = slider1.value;
output2.innerHTML = slider2.value;
var not_from_slider = true


function addWhat(params){
title_in_cluster = title_dict["Title_"+"cluster_" + params.nodes[0].toString()]
document.getElementById('what_content').textContent = title_in_cluster;
}

function addWho(params){
persons_in_cluster = person_dict["Person_"+"cluster_" + params.nodes[0].toString()]
var first_who_content = document.getElementById("first_who_content")
first_who_content.textContent = persons_in_cluster[0];
var who_content_div = document.createElement("div");
who_content_div.className = "dropdown-content";
for (k = 1; k < persons_in_cluster.length; k++){
    var paragraph = document.createElement("p");
    paragraph.textContent = persons_in_cluster[k];
    who_content_div.append(paragraph);
}
first_who_content.appendChild(who_content_div);
}

function addWhen(params){
date_in_cluster = date_dict["Date_"+"cluster_" + params.nodes[0].toString()]
var first_when_content = document.getElementById("first_when_content")
first_when_content.textContent = date_in_cluster[0];
var when_content_div = document.createElement("div");
when_content_div.className = "dropdown-content";
for (k = 1; k < date_in_cluster.length; k++){
    var paragraph = document.createElement("p");
    paragraph.textContent = date_in_cluster[k];
    when_content_div.append(paragraph);
}
first_when_content.appendChild(when_content_div);
}

function addPlace(params){
place_in_cluster = place_dict["Place_"+"cluster_" + params.nodes[0].toString()]
var first_where_place_content = document.getElementById("first_where_place_content")
first_where_place_content.textContent = place_in_cluster[0];
var where_place_content = document.createElement("div");
where_place_content.className = "dropdown-content";
for (k = 1; k < place_in_cluster.length; k++){
    var paragraph = document.createElement("p");
    paragraph.textContent = place_in_cluster[k];
    where_place_content.append(paragraph);
}
first_where_place_content.appendChild(where_place_content);
}

function addRelatedEvents(params){
var related_events_div = document.getElementById("related_events_div");
related_events_div.innerHTML = '';
related_events_in_cluster = related_events["cluster_" + params.nodes[0].toString()]
    if(related_events_in_cluster.length == 0){
        related_events_div.innerText = "No Related Events";
    }
    else{
        for (k = 0; k < related_events_in_cluster.length; k++){
            var newDiv = document.createElement('div');
            newDiv.className = "related_events"
            newDiv.cluster_name = related_events_in_cluster[k]
            newDiv.innerHTML = "Related Event " + (k+1).toString();
            newDiv.addEventListener('click', showRelatedEvent, false);
            related_events_div.appendChild(newDiv);
        }
    }
}

var showRelatedEvent = function() {
    var options_2 = {
    scale: 2,
    offset: { x: 0, y: 0},
    animation: {
      duration: 5000,
      easingFunction: "linear",
    },
  };
  var cluster_id = parseInt(this.cluster_name.replace("cluster_", ""));
  network.focus(cluster_id, options_2);
  network.selectNodes([cluster_id], true);
}


//function addCountry(){
//var first_where_country_content = document.getElementById("first_where_country_content")
//first_where_country_content.textContent = "Name Name";
//var where_country_content = document.createElement("div");
//where_country_content.className = "dropdown-content";
//for (k = 1; k < 10; k++){
//    var paragraph = document.createElement("p");
//    paragraph.textContent = k;
//    where_country_content.append(paragraph);
//}
//first_where_country_content.appendChild(where_country_content);
//}

//function addCity(){
//var first_where_city_content = document.getElementById("first_where_city_content")
//first_where_city_content.textContent = "Name Name";
//var where_city_content = document.createElement("div");
//where_city_content.className = "dropdown-content";
//for (k = 1; k < 10; k++){
//    var paragraph = document.createElement("p");
//    paragraph.textContent = k;
//    where_city_content.append(paragraph);
//}
//first_where_city_content.appendChild(where_city_content);
//}

function addSummary(params){
  document.getElementById('summary_button').style.display = "none";
  var python_url = 'http://127.0.0.1:5000/get_summary_for_cluster';
  var cluster_method_no = '?cluster_method_no=' + document.getElementById("cluster_method_list").value + ":" + "cluster_" + params.nodes[0].toString();
  $.ajax({
    url: python_url + cluster_method_no,
    type: 'GET',
    success: function(data){
         summary = data;
         document.getElementById('summary_button').style.display = "inline-block";
    }
    });

//summary_in_cluster = summary_dict["Summary_"+"cluster_" + params.nodes[0].toString()]
//document.getElementById('summary_content').textContent = summary_in_cluster;
}

function addNews(params){
docs_in_cluster = cluster_dict["cluster_" + params.nodes[0].toString()]
document.getElementById('newsArticlesText').innerHTML = '';
 for (k = 0; k < docs_in_cluster.length; k++){
            doc_no = docs_in_cluster[k]
            var newDiv = document.createElement('div');
            newDiv.fulltext = text_dict[doc_no];
            newDiv.title = docs_dict[doc_no];
            newDiv.id = 'doc_'+ doc_no;
            newDiv.className = 'cluster';
            newDiv.innerHTML = docs_dict[doc_no] + '<br/>'
            newDiv.addEventListener('click', openFullNews, false);
            document.getElementById('newsArticlesText').appendChild(newDiv);
        }
}



// Function for click event
function click(){
  network.on("click", function (params) {
    clicked_node = params.nodes[0]
    if (typeof clicked_node !== 'undefined'){
        addWhat(params);
        addWho(params);
        addWhen(params);
        addPlace(params);
        addNews(params);
        addSummary(params);
        addRelatedEvents(params);
    }});
}

// Function for displaying tree
function displayTree() {
set_entity_names();

if(document.getElementById("cluster_method_list").value == "Hubble"){
    news_path = '/results_dynamic/news.json';
}
else if (document.getElementById("cluster_method_list").value == "Voyager"){
    news_path = '/results_dynamic/top2vecnews.json';
}
    fetch(news_path).then(response => {
  return response.json();
}).then(data => {
  var n = data["nodes"]
  var e = data["edges"]
  cluster_dict = data["cluster_dict"]
  docs_dict = data["docs_dict"]
  text_dict = data["text_dict"]
  related_events = data["related_events"]
  title_dict = data["Title_dict"]
  summary_dict = data["Summary_dict"]
  place_dict = data["Place_dict"]
  person_dict = data["Person_dict"]
  date_dict = data["Date_dict"]
  possible_content_depth = data["possible_content_depth"]
  if(not_from_slider){
      document.getElementById("content_depth_number").max = possible_content_depth;
      set(document.getElementById("content_depth_number"), possible_content_depth);
    }
  not_from_slider = true
  nodes = new vis.DataSet(n);
  // create an array with edges
  edges = new vis.DataSet(e);
  // create a network
  var container = document.getElementById("hierarchicalStructure");
  var data = {
    nodes: nodes,
    edges: edges
  };
  var options = {

  nodes: { borderWidth: 1, color: {
            background: '#7CB9E8',
            border:  '#000000'
//            highlight: {
//                border: '#2B7CE9',
//                background: 'DarkSeaGreen'
//            }
            }},
  interaction: {
      hover: true,
//      tooltipDelay: 200,
//      hideEdgesOnDrag: true,
//      hideEdgesOnZoom: true,
    },

  layout: {
  hierarchical: {
   enabled: $('#hierarchy_display_checkbox').is(':checked'),
    direction: 'UD',
    nodeSpacing: 100,
    sortMethod : 'directed',
    levelSeparation: 300
  },
  },

};
  network = new vis.Network(container, data, options);
//  click(network, cluster_dict, docs_dict);
  click();
}).catch(err => {
  alert(err);
});
}

var openFullNews = function() {
     document.getElementById('hierarchicalStructure').style.filter = "blur(2px)";
     document.getElementById('displayInfo').style.filter = "blur(2px)";
     document.getElementById('mainHeader').style.filter = "blur(2px)";
     document.body.style.backgroundColor = "DimGrey";
     modal_news.style.display = "block";
     modal_news_content.innerHTML = this.fulltext;
     modal_news_header.innerHTML = this.title;
};


function showSummary(){
     document.getElementById('hierarchicalStructure').style.filter = "blur(2px)";
     document.getElementById('displayInfo').style.filter = "blur(2px)";
     document.getElementById('mainHeader').style.filter = "blur(2px)";
     document.body.style.backgroundColor = "DimGrey";
     modal_news.style.display = "block";
     modal_news_content.innerHTML = summary;
     modal_news_header.innerHTML = "Summary";
}




/*window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
        document.getElementById('hierarchicalStructure').style.filter = "blur(0)";
        document.getElementById('displayInfo').style.filter = "blur(0)";
    }
}*/

close_news.onclick = function() {
    modal_news.style.display = "none";
    document.getElementById('hierarchicalStructure').style.filter = "blur(0)";
    document.getElementById('displayInfo').style.filter = "blur(0)";
    document.getElementById('mainHeader').style.filter = "blur(0)";
    document.body.style.backgroundColor = "white";
}

close_settings.onclick = function() {
    modal_settings.style.display = "none";
    document.getElementById('hierarchicalStructure').style.filter = "blur(0)";
    document.getElementById('displayInfo').style.filter = "blur(0)";
    document.getElementById('mainHeader').style.filter = "blur(0)";
    document.body.style.backgroundColor = "white";
    set_entity_names();
}

function set_entity_names(){

//var entity_names_div = document.getElementById("checkboxId")
//entity_names_div.textContent = "";
//var checkedValues = $("li input[type=checkbox]:checked").map(function () {
//        return $(this).attr("id");
//    });
//
//for (z = 0; z < checkedValues.length; z++){
//    var entity_div = document.createElement("div");
//    entity_div.className = "entity_div";
//    entity_div.textContent = checkedValues[z];
//    entity_names_div.appendChild(entity_div);
//    }
}

function openSettings(){
     document.getElementById('hierarchicalStructure').style.filter = "blur(2px)";
     document.getElementById('displayInfo').style.filter = "blur(2px)";
     document.getElementById('mainHeader').style.filter = "blur(2px)";
     document.body.style.backgroundColor = "DimGrey";
     modal_settings.style.display = "block";
//     document.getElementById("tablinks_place_id").click();
}

function generateHierarchy(){
var python_url = 'http://127.0.0.1:5000/generate_hierarchy';
var split_entity_string = "";
$("input:checkbox[name=constraint]:checked").each(function () {
            split_entity_string = split_entity_string + $(this).attr("id") + ":";
        });
if (split_entity_string == "") {
    swal("Please select atleast one entity!", "Time, Place, Person, Content", "error");
    throw 'Stopping Execution';
}
split_entity_string = '?split_entity_string=' + split_entity_string.slice(0,-1);
var content_depth_needed = '&content_depth_needed=' + document.getElementById("content_depth_number").value;
var content_capture_needed = '&content_capture_needed=' + document.getElementById("content_capture_number").value;
var time_place_weight = '&time_place_weight=' + document.getElementById("demo1").innerText;
var content_weight = '&content_weight=' + document.getElementById("demo2").innerText;
var topic_interest_keyword = '&topic_interest_keyword=' + document.getElementById("topic_interest_keyword").value;
var from_date_keyword = '&from_date_keyword=' + document.getElementById("from_date_keyword").value;
var to_date_keyword = '&to_date_keyword=' + document.getElementById("to_date_keyword").value;
var cluster_method = '&cluster_method=' + document.getElementById("cluster_method_list").value;


setProgress();
$.ajax({
url: python_url + split_entity_string + content_depth_needed + content_capture_needed + time_place_weight
    + content_weight +topic_interest_keyword + from_date_keyword + to_date_keyword + cluster_method,
type: 'GET',
success: function(data){
    if(data == 'success'){
        displayTree();
        swal({
          title: "Generated Hierarchy",
          text: " ",
          icon: "success",
          buttons: false,
          timer: 1500,
          closeOnClickOutside: false
        })
//        swal.stopLoading();
//        swal.close();
        }
    else{
        swal("Error while generating hierarchy!", "", "error");
        }
    }
});
}


function set_content_depth(){
    set(document.getElementById("content_depth_number"), 1000);
}

function content_depth_slider_onchange(){
    not_from_slider = false
}

function cluster_method_change(){
    if(document.getElementById("cluster_method_list").value == "Hubble"){
        document.getElementById("settings").style.display = "block";
        reset_event_representation_news_content();
        }
    else if (document.getElementById("cluster_method_list").value == "Voyager"){
        document.getElementById("settings").style.display = "none";
        reset_event_representation_news_content();
        }
    displayTree();
}


function search_focus_node(){
  var options_2 = {
    scale: 2,
    offset: { x: 0, y: 0},
    animation: {
      duration: 5000,
      easingFunction: "linear",
    },
  };
  var python_url = 'http://127.0.0.1:5000/search_node';
  var search_term = '?search_term=' + document.getElementById("search_node").value;
  $.ajax({
    url: python_url + search_term,
    type: 'GET',
    success: function(data){
        if(typeof +data == "number"){
             network.focus(data, options_2);
             network.selectNodes([data], true);
            }
        else{
            swal({
          title: "Could not find a matching cluster",
          text: " ",
          icon: "error",
          buttons: false,
          timer: 2000,
          closeOnClickOutside: false
        })
            }
    }
});
}

function key_down(e) {
    if(e.keyCode == 13) {
      search_focus_node();
    }
  }

function openEntitySettings(evt, cityName) {
  var i, tabcontent
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  document.getElementById(cityName).style.display = "block";
}


slider1.oninput = function() {
  output1.innerHTML = this.value;
  output2.innerHTML = "";
  output2.innerHTML = (10 - (this.value)*10)/10;
  set(slider2, output2.innerHTML);
}
slider2.oninput = function() {
  output2.innerHTML = this.value;
  output1.innerHTML = "";
  output1.innerHTML = (10 - (this.value)*10)/10;
  set(slider1, output1.innerHTML);
}

function set(elm, val) {
    if (val) {
        elm.value = val;
    }
    elm.setAttribute('value', elm.value);
}



function reset_event_representation_news_content(){
document.getElementById("hierarchicalStructure").innerHTML = "";
document.getElementById("what_content").innerHTML = "Title";
document.getElementById("first_who_content").innerHTML = "Person";
document.getElementById("first_when_content").innerHTML = "Time";
document.getElementById("first_where_place_content").innerHTML = "Place";
document.getElementById("newsArticlesText").innerHTML = "";
document.getElementById("related_events_div").innerHTML = "Related Events";
//document.getElementById('summary_content').innerHTML = "Summary"
}


function setProgress() {
reset_event_representation_news_content();

//progress.style.textAlign = "center";
//progress.style.fontSize = "large";
swal({
  title: "Generating Hierarchy . . .",
  text: " ",
  icon: "info",
  buttons: false,
  closeOnClickOutside: false
})
}

function saveAndgenerateHierarchy(){
saveSettings();
set_entity_names();
generateHierarchy();
}

function restoreDefaults(){
output1.innerHTML = 0;
output2.innerHTML = 1;
set(slider1, output1.innerHTML)
set(slider2, output2.innerHTML)
//document.getElementById("content_depth_number").value = 2
}

function saveSettings(){
    modal_settings.style.display = "none";
    document.getElementById('hierarchicalStructure').style.filter = "blur(0)";
    document.getElementById('displayInfo').style.filter = "blur(0)";
    document.getElementById('mainHeader').style.filter = "blur(0)";
    document.body.style.backgroundColor = "white";
    set_entity_names();
}




// for dragging and dropping

const draggable_list = document.getElementById('draggable-list');
const entity_names = [
  'Time',
  'Content',
  'Place',
  'Person'
];
// Store listitems
const listItems = [];
let dragStartIndex;
createList();
// Insert list items into DOM
function createList() {
  [...entity_names]
    .forEach((entity, index) => {
      const listItem = document.createElement('li');
      listItem.setAttribute('data-index', index);
      if(entity == "Content"){
          listItem.innerHTML = `
            <div class="draggable" draggable="true">
              <input type="checkbox" id= ${entity} name="constraint" class="entity-name" checked/>
              <label for= ${entity} style="font-size: 18px">&nbsp &nbsp ${entity}</label>
            </div>
          `;
      }
      else{
          listItem.innerHTML = `
            <div class="draggable" draggable="true">
              <input type="checkbox" id= ${entity} name="constraint" class="entity-name"/>
              <label for= ${entity} style="font-size: 18px">&nbsp &nbsp ${entity}</label>
            </div>
          `;
      }
      listItems.push(listItem);
      draggable_list.appendChild(listItem);
    });
  addEventListeners();
}
function dragStart() {
  // console.log('Event: ', 'dragstart');
  dragStartIndex = +this.closest('li').getAttribute('data-index');
}
function dragEnter() {
  // console.log('Event: ', 'dragenter');
  this.classList.add('over');
}
function dragLeave() {
  // console.log('Event: ', 'dragleave');
  this.classList.remove('over');
}
function dragOver(e) {
  // console.log('Event: ', 'dragover');
  e.preventDefault();
}
function dragDrop() {
  // console.log('Event: ', 'drop');
  const dragEndIndex = +this.getAttribute('data-index');
  swapItems(dragStartIndex, dragEndIndex);
  this.classList.remove('over');
}

// Swap list items that are drag and drop
function swapItems(fromIndex, toIndex) {
  const itemOne = listItems[fromIndex].querySelector('.draggable');
  const itemTwo = listItems[toIndex].querySelector('.draggable');
  listItems[fromIndex].appendChild(itemTwo);
  listItems[toIndex].appendChild(itemOne);
}

function addEventListeners() {
  const draggables = document.querySelectorAll('.draggable');
  const dragListItems = document.querySelectorAll('.draggable-list li');
  draggables.forEach(draggable => {
    draggable.addEventListener('dragstart', dragStart);
  });
  dragListItems.forEach(item => {
    item.addEventListener('dragover', dragOver);
    item.addEventListener('drop', dragDrop);
    item.addEventListener('dragenter', dragEnter);
    item.addEventListener('dragleave', dragLeave);
  });
}



