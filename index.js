const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const paramConfig = new ParamConfig(
  "./config.json",
  document.querySelector("#cfg-outer")
);
paramConfig.addCopyToClipboardHandler("#share-btn");

ctx.fillStyle = "black";
ctx.strokeStyle = "white";

function hexToRGB(hex) {
  const match = hex
    .toUpperCase()
    .match(/^#?([\dA-F]{2})([\dA-F]{2})([\dA-F]{2})$/);
  if (!match) return false;
  return [
    parseInt(match[1], 16),
    parseInt(match[2], 16),
    parseInt(match[3], 16),
  ];
}

let mandela;

function generate() {
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const sideLength = paramConfig.getVal("resolution");
  const shape = [sideLength, sideLength, 1];
  const segmentAxioms = [];
  const colours = [];
  for (let i = 0; i < paramConfig.getVal("num-axioms"); i++) {
    segmentAxioms.push({
      x: (Math.random() * sideLength) / 2,
      y: (Math.random() * sideLength) / 2,
      weight:
        0.5 -
        paramConfig.getVal("weight-influence") / 2 +
        Math.random() * paramConfig.getVal("weight-influence"),
    });
    colours.push([Math.random(), Math.random(), Math.random()]);
  }
  colours.push(
    hexToRGB(paramConfig.getVal("background-colour")).map((n) => n / 255)
  );

  const aspectRatio = canvas.width / canvas.height;

  if (mandela != null) {
    mandela.dispose();
  }

  mandela = tf.tidy(() => {
    const xCoords = tf.range(0, sideLength).tile([sideLength]).reshape(shape);
    const yCoords = xCoords.transpose().reshape(shape);
    // [..., [segmentIndex, score]]
    let segments = tf
      .stack([tf.fill(shape, 0), tf.fill(shape, Infinity)], 2)
      .reshape([sideLength, sideLength, 2]);
    for (let i = 0; i < segmentAxioms.length; i++) {
      const currentScore = xCoords
        .sub(segmentAxioms[i].x)
        .square()
        .add(yCoords.sub(segmentAxioms[i].y).square())
        .mul(segmentAxioms[i].weight);
      const scores = segments.slice([0, 0, 1], shape);
      segments = tf
        .stack(
          [
            tf.where(
              currentScore.less(scores),
              tf.fill(shape, i),
              segments.slice([0, 0, 0], shape)
            ),
            scores.minimum(currentScore),
          ],
          2
        )
        .reshape(segments.shape);
    }

    const centerXCoords = xCoords.sub(sideLength / 2).abs();
    const centerYCoords = yCoords.sub(sideLength / 2).abs();

    const segmented = segments.slice([0, 0, 0], shape);
    const cutOutSegmented = segmented.where(
      centerXCoords
        .square()
        .add(centerYCoords.square())
        .less((sideLength / 2) ** 2),
      tf.fill(shape, segmentAxioms.length)
    );

    const minCoords = tf.where(
      centerXCoords.less(centerYCoords),
      centerXCoords,
      centerYCoords
    );
    const maxCoords = tf.where(
      centerXCoords.greater(centerYCoords),
      centerXCoords,
      centerYCoords
    );

    const symmetricalSegments = cutOutSegmented
      .reshape([-1])
      .gather(
        minCoords
          .sub(sideLength / 2)
          .abs()
          .add(
            maxCoords
              .sub(sideLength / 2)
              .abs()
              .mul(sideLength)
          )
          .cast("int32")
      )
      .reshape(shape);

    let segmentsWithAspectRatio;

    if (aspectRatio < 1) {
      let wallPadding = tf.fill(
        [Math.floor(((1 - aspectRatio) * sideLength) / 2), sideLength, 1],
        segmentAxioms.length
      );
      segmentsWithAspectRatio = wallPadding.concat(
        symmetricalSegments.concat(wallPadding, 1),
        0
      );
    } else {
      let floorCeilingPadding = tf.fill(
        [sideLength, Math.floor(((aspectRatio - 1) * sideLength) / 2), 1],
        segmentAxioms.length
      );
      segmentsWithAspectRatio = floorCeilingPadding.concat(
        symmetricalSegments.concat(floorCeilingPadding, 1),
        1
      );
    }

    let mandela;
    if (paramConfig.getVal("use-colours")) {
      let pixelData = tf.zeros([
        ...segmentsWithAspectRatio.shape.slice(0, 2),
        3,
      ]);
      for (let i = 0; i < colours.length; i++) {
        pixelData = pixelData.add(
          segmentsWithAspectRatio.equal(i).mul(colours[i])
        );
      }
      mandela = pixelData;
    } else {
      const edges = segmentsWithAspectRatio
        .conv2d(
          tf
            .tensor2d([
              [0, 1, 0],
              [1, 1, 1],
              [0, 1, 0],
            ])
            .reshape([3, 3, 1, 1]),
          1,
          1
        )
        .notEqual(segmentsWithAspectRatio.mul(5));

      mandela = tf
        .zeros([3])
        .where(edges, tf.tensor(colours[colours.length - 1]));
    }
    return tf.keep(mandela);
  });

  tf.browser.toPixels(mandela, canvas);
}

window.onresize = (evt) => {
  canvas.width = $("#canvas").width();
  canvas.height = $("#canvas").height();
  generate();
};

paramConfig.addListener(generate, ["generate"]);

paramConfig.onLoad(window.onresize);
