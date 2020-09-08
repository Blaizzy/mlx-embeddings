# Diagram Guidelines

## Preferred file format

For complex diagrams, use the `.drawio.svg` format.

Files with the `.drawio.svg` extension are SVG files with embedded [draw.io](https://www.diagrams.net/) source code. Using that format lends itself to a developer-friendly workflow: it is valid SVG, plays well with `git diff` and can be edited in lock-step using various online and offline flavours of draw.io. If you use VS Code, you can use an [extension](https://marketplace.visualstudio.com/items?itemName=hediet.vscode-drawio) for draw.io integration.

Files in the `.drawio.svg` format can be processed offline.

## Embedding a diagram into Markdown

To embed a `.drawio.svg` file into Markdown, use the same syntax as for any image. Example: `![My diagram](my-diagram.drawio.svg)`

Mind that GitHub doesn’t allow styling in Markdown documents. Where styling is allowed (e.g. in the exported brew.sh version of the documentation), always set a background colour of `white` for the diagram. That’s the colour draw.io assumes, and keeps the diagram easy to read in dark mode without further customization. You can use the CSS selector `img[src$=".drawio.svg"]` for styling.

## Example

Example for an SVG image embedded into Markdown:

```md
![Example diagram: Managing Pull Requests](assets/img/docs/managing-pull-requests.drawio.svg)
```

Result:

![Example diagram: Managing Pull Requests](assets/img/docs/managing-pull-requests.drawio.svg)

Example for styling (where allowed):

```css
img[src$=".drawio.svg"] {
  background-color: white;
  margin-bottom: 20px;
  padding: 5%;
  width: 90%;
}

@media (prefers-color-scheme: dark) {
  img[src$=".drawio.svg"] {
    filter: invert(85%);
    -webkit-filter: invert(85%);
  }
}
```
