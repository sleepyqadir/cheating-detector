const electron = require("electron");
const path = require("path");
const url = require("url");

const { app, BrowserWindow, Menu, ipcMain } = electron;

let mainWindow;

app.on("ready", function() {
  // Create new window
  mainWindow = new BrowserWindow({});
  // Load html in window
  mainWindow.loadURL(
    url.format({
      pathname: path.join(__dirname, "mainWindow.html"),
      protocol: "file:",
      slashes: true
    })
  );
});
