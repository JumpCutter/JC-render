var isExampleView = false;
function setExample() {
    var setClass = isExampleView ? '' : ' example-view';
    var body = document.querySelector('body');
    var exampleBlocks = document.querySelectorAll('.exampleblock');
    var tables = document.querySelectorAll('table');

    body.className = 'body article toc2 toc-left' + setClass;

    tables.forEach(function(t) {
        t.parentNode.style.overflowX = "scroll";
        console.log(t.parentNode);
    });

    exampleBlocks.forEach(function(exampleElem) {
        console.log(exampleElem);
        exampleElem.className = 'exampleblock' + setClass;
    });
    isExampleView = !isExampleView;
}


var isDarkMode = true;
var lightThemeUrl = 'https://jmblog.github.io/color-themes-for-google-code-prettify/themes/tomorrow.min.css';
var darkThemeUrl = 'https://jmblog.github.io/color-themes-for-google-code-prettify/themes/atelier-savanna-dark.min.css';
function changeTheme() {
    if (isDarkMode) {
        document.body.innerHTML += `
        <link rel="stylesheet" href="${lightThemeUrl}">`;
    } else {
        document.body.innerHTML += `
        <link rel="stylesheet" href="${darkThemeUrl}">`;
    }
    isDarkMode = !isDarkMode;
    runFooter()
}

function runFooter() {
    // var aAPI = document.querySelector('a[href="#api"]');
    // if (aAPI) {
    //     aAPI.href = './project.html';
    // }
    var apiModeButtons = document.querySelectorAll('.button.api_mode > a');
    apiModeButtons.forEach(function(b) {
        b.onclick = function() {
            setExample();
        };
    });
    var themeButtons = document.querySelectorAll('.button.theme_mode > a');
    themeButtons.forEach(function(b) {
        b.innerHTML = isDarkMode ? 'Dark Mode' : 'Light Mode'
        b.onclick = function() {
            changeTheme();
        };
    });
}
