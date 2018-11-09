/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnetexamples.infer.javaapi.predictor;

import org.apache.mxnet.infer.javaapi.ObjectDetector;
import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.DataDesc;
import org.apache.mxnet.javaapi.Shape;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.List;

public class PredictorExample {
    @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefix = "/model/ssd_resnet50_512";
    @Option(name = "--input-image", usage = "the input image")
    private String inputImagePath = "/images/dog.jpg";
    @Option(name = "--input-dir", usage = "the input batch of images directory")
    private String inputImageDir = "/images/";

    final static Logger logger = LoggerFactory.getLogger(PredictorExample.class);

    public static BufferedImage reshapeImage(BufferedImage buf, int newWidth, int newHeight) {
        BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(buf, 0, 0, newWidth, newHeight, null);
        g.dispose();
        return resizedImage;
    }

    public static float[] imagePreprocess(BufferedImage buf) {
        // Get height and width of the image
        int w = buf.getWidth();
        int h = buf.getHeight();

        // get an array of integer pixels in the default RGB color mode
        int[] pixels = buf.getRGB(0, 0, w, h, null, 0, w);

        // 3 times height and width for R,G,B channels
        float[] result = new float[3 * h * w];

        int row = 0;
        // copy pixels to array vertically
        while (row < h) {
            int col = 0;
            // copy pixels to array horizontally
            while (col < w) {
                int rgb = pixels[row * w + col];
                // getting red color
                result[0 * h * w + row * w + col] = (rgb >> 16) & 0xFF;
                // getting green color
                result[1 * h * w + row * w + col] = (rgb >> 8) & 0xFF;
                // getting blue color
                result[2 * h * w + row * w + col] = rgb & 0xFF;
                col += 1;
            }
            row += 1;
        }
        buf.flush();
        return result;
    }

    public static void main(String[] args) {
        PredictorExample inst = new PredictorExample();
        CmdLineParser parser  = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            parser.printUsage(System.err);
            System.exit(1);
        }
        // Prepare the model
        List<Context> context = new ArrayList<Context>();
        if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
                Integer.valueOf(System.getenv("SCALA_TEST_ON_GPU")) == 1) {
            context.add(Context.gpu());
        } else {
            context.add(Context.cpu());
        }
        List<DataDesc> inputDesc = new ArrayList<>();
        Shape inputShape = new Shape(new int[]{1, 3, 224, 224});
        inputDesc.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
        Predictor predictor = new Predictor(inst.modelPathPrefix, inputDesc, context,0);
        // Prepare data
        BufferedImage img = ObjectDetector.loadImageFromFile(inst.inputImagePath);
        img = reshapeImage(img, 224, 224);
        // predict
        float[][] result = predictor.predict(new float[][]{imagePreprocess(img)});
    }

}
