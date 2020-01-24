const listElements = document.querySelector(".posts");
const template = document.querySelector("#single-post");
const listposts = document.querySelector("ul");
const toBase64 = file =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });

async function Main() {
  const file = document.querySelector("#myfile").files;
  const imageLIst = [];
  for (var i = 0; i <= file.length - 1; i++) {
    const base64_result = await toBase64(file[i]);
    var strImage = base64_result.replace(/^data:.*?;base64,/, "");
    imageLIst.push(strImage);
  }
  console.log(imageLIst);
  sendPost(imageLIst);
  // document.getElementById("result").innerHTML = strImage;
}
const xhr = new XMLHttpRequest(("Content-type", "application/json"));

xhr.responseType = "json";

function sendHttpRequest(method, url, json) {
  const promise = new Promise((resolve, reject) => {
    xhr.open(method, url);
    console.log(json);
    xhr.onload = function(params) {
      // const posts = JSON.parse(xhr.response)
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response);
      } else {
        reject(new Error("something went wrong..."));
      }
    };
    xhr.onerror = function(params) {
      reject(new Error("something went wrong..."));
    };
    xhr.send(JSON.stringify(json));
  });
  return promise;
}
async function sendPost(body) {
  console.log(body);
  const json = {
    docs: body
  };
  console.log(json);
  const result = await sendHttpRequest(
    "POST",
    "http://c3551dd3.ngrok.io/word-text/123",
    json
  );
  for (let i in result.text) {
    createNode(result.text[i]);
  }
  console.log(result);
}
function Check() {
  const list = [];
  console.log("Iterate with for...of:");
  for (const li of document.querySelectorAll(".mb-57")) {
    console.log(li);
    list.push(li.textContent);
  }
  console.log(list);
  similarity(list);
}

async function similarity(body) {
  console.log(body);
  const json = {
    blogs: body
  };
  console.log(json);
  const result = await sendHttpRequest(
    "POST",
    "http://2831d247.ngrok.io/similarity/123",
    json
  );
  let final_result = [];
  console.log(result.length);

  for (let [key, value] of Object.entries(result["data"])) {
    //let simililar = result[i]
    console.log(key);

    let x = value.pop();
    console.log(x);

    if (x[0] == key) {
      x = value.pop();
      console.log(x);
    }
    final_result.push(x);

    //final_result.push(simililar)
  }

  final_result.pop();
  console.log(final_result);

  var ctxB = document.getElementById("barChart").getContext("2d");

  const labels = [];
  const data = [];
  const backgroundColor = [];
  const borderColor = [];
  let j = 0;
  for (let i of final_result) {
    if (i[1] * 100 >= 95) {
      data.push(i[1]);
      backgroundColor.push("rgba(255, 99, 132, 0.2)");
      borderColor.push("rgba(255,99,132,1)");
      labels.push(j + " similarity with " + i[0]);
    } else {
      data.push(i[1]);
      backgroundColor.push("rgba(54, 162, 235, 0.2)");
      borderColor.push("rgba(54, 162, 235, 1)");
      labels.push(j + " similarity with " + i[0]);
    }
    j++;
  }
  var myBarChart = new Chart(ctxB, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "# of Similarity",
          data,
          backgroundColor,
          borderColor,
          borderWidth: 1
        }
      ]
    },
    options: {
      scales: {
        yAxes: [
          {
            ticks: {
              beginAtZero: true
            }
          }
        ]
      }
    }
  });
  console.log(result);
  scroll(0, 0);
}

function createNode(post) {
  postEl = document.importNode(template.content, true);
  postEl.querySelector("p").textContent = post;
  listElements.appendChild(postEl);
}
