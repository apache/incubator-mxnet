package org.apache.mxnet.internal.c_api;

import org.junit.Test;
import static org.apache.mxnet.internal.c_api.global.mxnet.*;

public class UnitTest {
    @Test public void test() {
        int[] version = new int[1];
        MXGetVersion(version);
        System.out.println(version[0]);
    }
}
