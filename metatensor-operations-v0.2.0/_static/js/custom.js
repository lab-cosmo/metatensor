(() => {

// pick the right tab from the url hash fragment, if any
//
// There is an issue for this on sphinx-design, but without much progress
// https://github.com/executablebooks/sphinx-design/issues/68#issuecomment-1270701806
function pick_tab_from_url() {
    const hash = window.location.hash;
    if (hash === "") {
        return;
    }

    const label = document.querySelector(hash);
    if (label === null || !label.classList.contains("sd-tab-label")) {
        return;
    }

    label.control.checked = true;
}

document.addEventListener("DOMContentLoaded", pick_tab_from_url);
window.addEventListener("hashchange", pick_tab_from_url);


// listen to tab change & update url accordingly
function set_url_from_tab() {
    for (const input of document.querySelectorAll(".sd-tab-set input")) {
        input.addEventListener("change", () => {
            const label = document.querySelector(`label[for="${input.id}"`);
            window.location.hash = `#${label.id}`
        });
    }
}

document.addEventListener("DOMContentLoaded", set_url_from_tab);

})();
