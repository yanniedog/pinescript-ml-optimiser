(() => {
  function bindRightClick() {
    const container = document.getElementById("objective-graph");
    if (!container) {
      return;
    }
    const graphDiv = container.querySelector(".js-plotly-plot");
    if (!graphDiv || graphDiv.__rightclick_bound) {
      return;
    }
    graphDiv.__rightclick_bound = true;

    graphDiv.addEventListener("plotly_hover", (event) => {
      if (event && event.points && event.points.length) {
        const point = event.points[0];
        const label = point.customdata || (point.data && point.data.name);
        if (label) {
          graphDiv.dataset.lastLabel = String(label);
        }
      }
    });

    graphDiv.addEventListener("plotly_click", (event) => {
      if (!event || !event.points || !event.points.length) {
        return;
      }
      const point = event.points[0];
      const label = point.customdata || (point.data && point.data.name);
      if (!label) {
        return;
      }
      const button = document.querySelector(`#legend-box [data-label="${CSS.escape(label)}"]`);
      if (!button) {
        return;
      }
      button.click();
    });
  }

  setInterval(bindRightClick, 750);
})();
