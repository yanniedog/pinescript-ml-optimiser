import os

def generate_pine_scripts():
    # Create directory
    output_dir = "pinescripts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dictionary of 100 Indicators
    # Format: [Filename, Title, Overlay(True/False), InputsList, LogicString]
    # InputsList: (Name, Type, Default, Min, Max[, Title]) -> Type: 0=int, 1=float, 2=symbol
    
    indicators = [
        # 1-10
        ["RSI", "Relative Strength Index", False, [("len", 0, 14, 1, 100), ("ob", 0, 70, 50, 95), ("os", 0, 30, 5, 50)], 
         "r = ta.rsi(close, len)\nplot(r, title='RSI', color=color.purple)\nhline(ob, 'Overbought', color=color.red)\nhline(os, 'Oversold', color=color.green)"],
         
        ["MACD", "MACD", False, [("fast", 0, 12, 1, 50), ("slow", 0, 26, 1, 100), ("sig", 0, 9, 1, 50)], 
         "[m, s, h] = ta.macd(close, fast, slow, sig)\nplot(m, 'MACD', color=color.blue)\nplot(s, 'Signal', color=color.orange)\nplot(h, 'Hist', color=color.gray, style=plot.style_histogram)"],
         
        ["BollingerBands", "Bollinger Bands", True, [("len", 0, 20, 1, 100), ("mult", 1, 2.0, 1, 5)], 
         "[m, u, l] = ta.bb(close, len, mult)\nplot(m, 'Basis', color=color.orange)\nplot(u, 'Upper', color=color.blue)\nplot(l, 'Lower', color=color.blue)"],
         
        ["Stochastic", "Stochastic Oscillator", False, [("k", 0, 14, 1, 100), ("d", 0, 3, 1, 50), ("s", 0, 3, 1, 50)], 
         "stoch = ta.stoch(close, high, low, k)\nkl = ta.sma(stoch, s)\ndl = ta.sma(kl, d)\nplot(kl, 'K', color=color.blue)\nplot(dl, 'D', color=color.orange)\nhline(80)\nhline(20)"],
         
        ["ADX", "Average Directional Index", False, [("len", 0, 14, 1, 100), ("th", 0, 25, 1, 50)], 
         "[dp, dm, adx] = ta.dmi(len, len)\nplot(adx, 'ADX', color=color.red)\nhline(th, 'Threshold', color=color.gray)"],
         
        ["Ichimoku", "Ichimoku Cloud", True, [("ten", 0, 9, 1, 50), ("kij", 0, 26, 1, 100)], 
         "avg(l) => (ta.highest(high, l) + ta.lowest(low, l)) / 2\nt = avg(ten)\nk = avg(kij)\nplot(t, 'Tenkan', color=color.red)\nplot(k, 'Kijun', color=color.blue)"],
         
        ["SAR", "Parabolic SAR", True, [("start", 1, 0.02, 0, 1), ("inc", 1, 0.02, 0, 1), ("max", 1, 0.2, 0, 1)], 
         "out = ta.sar(start, inc, max)\nplot(out, 'SAR', style=plot.style_cross, color=color.blue)"],
         
        ["Supertrend", "Supertrend", True, [("fac", 1, 3.0, 1, 10), ("len", 0, 10, 1, 100)], 
         "[st, dir] = ta.supertrend(fac, len)\nplot(st, 'Supertrend', color = dir < 0 ? color.green : color.red)"],
         
        ["WilliamsR", "Williams %R", False, [("len", 0, 14, 1, 100), ("ob", 0, -20, -100, 0), ("os", 0, -80, -100, 0)], 
         "w = ta.wpr(len)\nplot(w, '%R', color=color.purple)\nhline(ob)\nhline(os)"],
         
        ["MFI", "Money Flow Index", False, [("len", 0, 14, 1, 100), ("ob", 0, 80, 50, 100), ("os", 0, 20, 0, 50)], 
         "m = ta.mfi(close, len)\nplot(m, 'MFI', color=color.blue)\nhline(ob)\nhline(os)"],

        # 11-20
        ["Aroon", "Aroon", False, [("len", 0, 14, 1, 100)], 
         "[up, down] = ta.aroon(len)\nplot(up, 'Up', color=color.green)\nplot(down, 'Down', color=color.red)"],
         
        ["Keltner", "Keltner Channels", True, [("len", 0, 20, 1, 100), ("mult", 1, 2.0, 1, 10)], 
         "[m, u, l] = ta.kc(close, len, mult)\nplot(m, color=color.gray)\nplot(u, color=color.blue)\nplot(l, color=color.blue)"],
         
        ["Donchian", "Donchian Channels", True, [("len", 0, 20, 1, 100)], 
         "u = ta.highest(high, len)\nl = ta.lowest(low, len)\nplot(u, 'Upper', color=color.green)\nplot(l, 'Lower', color=color.red)"],
         
        ["Vortex", "Vortex Indicator", False, [("len", 0, 14, 1, 100)], 
         "[vp, vm] = ta.vortex(len)\nplot(vp, 'VI+', color=color.green)\nplot(vm, 'VI-', color=color.red)"],
         
        ["TRIX", "TRIX", False, [("len", 0, 18, 1, 100)], 
         "m = ta.ema(ta.ema(ta.ema(close, len), len), len)\nout = 10000 * (m - m[1]) / m[1]\nplot(out, 'TRIX', color=color.blue)\nhline(0)"],
         
        ["AO", "Awesome Oscillator", False, [("f", 0, 5, 1, 50), ("s", 0, 34, 1, 100)], 
         "ao = ta.sma(hl2, f) - ta.sma(hl2, s)\nplot(ao, 'AO', style=plot.style_histogram, color=(ao >= 0 ? color.green : color.red))"],
         
        ["CCI", "Commodity Channel Index", False, [("len", 0, 20, 1, 100), ("lev", 0, 100, 1, 200)], 
         "c = ta.cci(close, len)\nplot(c, 'CCI', color=color.orange)\nhline(lev)\nhline(-lev)"],
         
        ["CMF", "Chaikin Money Flow", False, [("len", 0, 20, 1, 100)], 
         "c = ta.cmf(close, len)\nplot(c, 'CMF', color=color.green)\nhline(0)"],
         
        ["ChaikinOsc", "Chaikin Oscillator", False, [("f", 0, 3, 1, 20), ("s", 0, 10, 1, 50)], 
         "adl = ta.accdist()\nosc = ta.ema(adl, f) - ta.ema(adl, s)\nplot(osc, \"Chaikin Osc\", color=color.blue)\nhline(0)"],
         
        ["KST", "Know Sure Thing", False, [("r1", 0, 10, 1, 50), ("r2", 0, 15, 1, 50), ("r3", 0, 20, 1, 50), ("r4", 0, 30, 1, 50)], 
         "rc(l) => ta.roc(close, l)\nkst = (rc(r1)*1) + (rc(r2)*2) + (rc(r3)*3) + (rc(r4)*4)\nsig = ta.sma(kst, 9)\nplot(kst, 'KST', color=color.green)\nplot(sig, 'Signal', color=color.red)"],

        # 21-30
        ["UltOsc", "Ultimate Oscillator", False, [("l1", 0, 7, 1, 20), ("l2", 0, 14, 1, 40), ("l3", 0, 28, 1, 80)], 
         "u = ta.ultosc(l1, l2, l3)\nplot(u, 'Ultimate', color=color.purple)\nhline(70)\nhline(30)"],
         
        ["StochRSI", "Stochastic RSI", False, [("len", 0, 14, 1, 100), ("k", 0, 3, 1, 20), ("d", 0, 3, 1, 20)], 
         "r = ta.rsi(close, len)\nst = ta.stoch(r, r, r, len)\nK = ta.sma(st, k)\nD = ta.sma(K, d)\nplot(K, 'K', color=color.blue)\nplot(D, 'D', color=color.orange)"],
         
        ["TSI", "True Strength Index", False, [("long", 0, 25, 1, 100), ("short", 0, 13, 1, 50)], 
         "t = ta.tsi(close, long, short)\nplot(t, 'TSI', color=color.red)\nhline(0)"],
         
        ["Fisher", "Fisher Transform", False, [("len", 0, 9, 1, 50)], 
         "h = ta.highest(high, len)\nl = ta.lowest(low, len)\nvar v = 0.0\nv := 0.66 * ((hlc3 - l) / math.max(h - l, 0.001) - 0.5) + 0.67 * nz(v[1])\nv := v > 0.99 ? 0.999 : v < -0.99 ? -0.999 : v\nf = 0.5 * math.log((1 + v) / (1 - v))\nplot(f, \"Fisher\", color=color.blue)\nplot(nz(f[1]), \"Trigger\", color=color.orange)"],
         
        ["EOM", "Ease of Movement", False, [("len", 0, 14, 1, 100), ("div", 0, 10000, 1, 100000)], 
         "dm = ((high + low)/2) - ((high[1] + low[1])/2)\nbr = (volume / div) / (high - low == 0 ? 1 : high - low)\ne = ta.sma(dm / br, len)\nplot(e, \"EOM\", color=color.blue)\nhline(0)"],
         
        ["MassIndex", "Mass Index", False, [("len", 0, 9, 1, 50), ("sum", 0, 25, 1, 50)], 
         "r = high - low\ne1 = ta.ema(r, len)\ne2 = ta.ema(e1, len)\nrat = e1 / (e2 == 0 ? 1 : e2)\nm = math.sum(rat, sum)\nplot(m, \"Mass Index\", color=color.purple)\nhline(27)"],
         
        ["Chop", "Choppiness Index", False, [("len", 0, 14, 1, 100)], 
         "tr_sum = math.sum(ta.tr(), len)\nrange = ta.highest(high, len) - ta.lowest(low, len)\nci = 100 * math.log10(tr_sum / (range == 0 ? 1 : range)) / math.log10(len)\nplot(ci, \"Chop\", color=color.blue)\nhline(61.8)\nhline(38.2)"],
         
        ["DPO", "Detrended Price Oscillator", False, [("len", 0, 21, 1, 100)], 
         "d = close - ta.sma(close, len)[math.floor(len/2) + 1]\nplot(d, 'DPO', color=color.orange)\nhline(0)"],
         
        ["Force", "Force Index", False, [("len", 0, 13, 1, 100)], 
         "f = ta.ema(ta.change(close) * volume, len)\nplot(f, 'Force', color=color.blue)\nhline(0)"],
         
        ["Envelope", "Envelope", True, [("len", 0, 20, 1, 100), ("pct", 1, 5.0, 1, 20)], 
         "b = ta.sma(close, len)\nu = b + b * pct / 100\nl = b - b * pct / 100\nplot(b, 'Base', color=color.gray)\nplot(u, 'Upper', color=color.blue)\nplot(l, 'Lower', color=color.blue)"],

        # 31-40
        ["Alligator", "Alligator", True, [("j", 0, 13, 1, 50), ("t", 0, 8, 1, 30), ("l", 0, 5, 1, 20)], 
         "jaw = ta.sma(hl2, j)\nteeth = ta.sma(hl2, t)\nlips = ta.sma(hl2, l)\nplot(jaw[8], 'Jaw', color=color.blue)\nplot(teeth[5], 'Teeth', color=color.red)\nplot(lips[3], 'Lips', color=color.green)"],
         
        ["Accelerator", "Accelerator Oscillator", False, [("f", 0, 5, 1, 20), ("s", 0, 34, 1, 100)], 
         "ao = ta.sma(hl2, f) - ta.sma(hl2, s)\nac = ao - ta.sma(ao, 5)\nplot(ac, 'AC', style=plot.style_histogram, color=(ac > ac[1] ? color.green : color.red))"],
         
        ["Gator", "Gator Oscillator", False, [("j", 0, 13, 1, 50), ("t", 0, 8, 1, 30), ("l", 0, 5, 1, 20)], 
         "jaw = ta.sma(hl2, j)[8]\nteeth = ta.sma(hl2, t)[5]\nlips = ta.sma(hl2, l)[3]\nu = math.abs(jaw - teeth)\nd = -math.abs(teeth - lips)\nplot(u, 'Up', style=plot.style_histogram, color=color.green)\nplot(d, 'Down', style=plot.style_histogram, color=color.red)"],
         
        ["PVT", "Price Volume Trend", False, [], 
         "var pvt = 0.0\npvt := nz(pvt[1]) + (ta.change(close) / (close[1] == 0 ? 1 : close[1])) * volume\nplot(pvt, 'PVT', color=color.blue)"],
         
        ["ROC", "Rate of Change", False, [("len", 0, 9, 1, 100)], 
         "roc = ta.roc(close, len)\nplot(roc, 'ROC', color=color.blue)\nhline(0)"],
         
        ["Coppock", "Coppock Curve", False, [("l1", 0, 14, 1, 50), ("l2", 0, 11, 1, 50), ("smooth", 0, 10, 1, 50)], 
         "x = ta.roc(close, l1) + ta.roc(close, l2)\nc = ta.wma(x, smooth)\nplot(c, 'Coppock', color=color.purple)\nhline(0)"],
         
        ["VolOsc", "Volume Oscillator", False, [("s", 0, 5, 1, 20), ("l", 0, 10, 1, 50)], 
         "v = (ta.ema(volume, s) - ta.ema(volume, l)) / (ta.ema(volume, l) == 0 ? 1 : ta.ema(volume, l)) * 100\nplot(v, \"Vol Osc\", color=color.blue)\nhline(0)"],
         
        ["LinRegCh", "Linear Regression Channel", True, [("len", 0, 100, 1, 200), ("dev", 1, 2.0, 1, 5)], 
         "l = ta.linreg(close, len, 0)\ns = ta.stdev(close, len)\nplot(l, 'Mid')\nplot(l + s*dev, 'Upper')\nplot(l - s*dev, 'Lower')"],
         
        ["StdErrBands", "Standard Error Bands", True, [("len", 0, 20, 1, 100), ("mult", 1, 2.0, 1, 5)], 
         "l = ta.linreg(close, len, 0)\nse = ta.stdev(close, len) / math.sqrt(len)\nplot(l, 'LinReg', color=color.gray)\nplot(l + se*mult, 'Up', color=color.green)\nplot(l - se*mult, 'Dn', color=color.red)"],
         
        ["HMA", "Hull Moving Average", True, [("len", 0, 9, 1, 100)], 
         "h = ta.hma(close, len)\nplot(h, 'HMA', color=color.purple)"],

        # 41-50
        ["KAMA", "Kaufman Adaptive MA", True, [("len", 0, 10, 1, 50), ("f", 0, 2, 1, 10), ("s", 0, 30, 1, 100)], 
         "sum_chg = math.sum(math.abs(ta.change(close)), len)\ner = math.abs(ta.change(close, len)) / (sum_chg == 0 ? 1 : sum_chg)\nsc = math.pow(er * (2/(f+1) - 2/(s+1)) + 2/(s+1), 2)\nvar kama = 0.0\nkama := nz(kama[1], close) + sc * (close - nz(kama[1], close))\nplot(kama, \"KAMA\", color=color.blue)"],
         
        ["McGinley", "McGinley Dynamic", True, [("len", 0, 14, 1, 100)], 
         "var mg = 0.0\nprev_mg = nz(mg[1], close)\nratio = prev_mg == 0 ? 1 : close / prev_mg\ndiv = len * math.pow(ratio, 4)\nmg := prev_mg + (close - prev_mg) / (div == 0 ? 1 : div)\nplot(mg, \"McGinley\", color=color.orange)"],
         
        ["Pivots", "Pivot Points High Low", True, [("len", 0, 10, 1, 50)], 
         "ph = ta.pivothigh(high, len, len)\npl = ta.pivotlow(low, len, len)\nplot(ph, 'High', style=plot.style_cross, color=color.red, offset=-len)\nplot(pl, 'Low', style=plot.style_cross, color=color.green, offset=-len)"],
         
        ["PPO", "Percentage Price Oscillator", False, [("s", 0, 12, 1, 50), ("l", 0, 26, 1, 100), ("sig", 0, 9, 1, 50)], 
         "short = ta.ema(close, s)\nlong = ta.ema(close, l)\nppo = (short - long) / (long == 0 ? 1 : long) * 100\nsignal = ta.ema(ppo, sig)\nplot(ppo, \"PPO\", color=color.blue)\nplot(signal, \"Signal\", color=color.orange)"],
         
        ["ElderRay", "Elder Ray Index", False, [("len", 0, 13, 1, 50)], 
         "ema = ta.ema(close, len)\nbull = high - ema\nbear = low - ema\nplot(bull, 'Bull', color=color.green)\nplot(bear, 'Bear', color=color.red)\nhline(0)"],
         
        ["CMO", "Chande Momentum Oscillator", False, [("len", 0, 14, 1, 100)], 
         "cmo = ta.cmo(close, len)\nplot(cmo, 'CMO', color=color.blue)\nhline(50)\nhline(-50)"],
         
        ["DeMarker", "DeMarker", False, [("len", 0, 14, 1, 100)], 
         "dmax = high > high[1] ? high - high[1] : 0\ndmin = low < low[1] ? low[1] - low : 0\nsum = ta.sma(dmax, len) + ta.sma(dmin, len)\nde = ta.sma(dmax, len) / (sum == 0 ? 1 : sum)\nplot(de, \"DeMarker\", color=color.blue)\nhline(0.7)\nhline(0.3)"],
         
        ["VHF", "Vertical Horizontal Filter", False, [("len", 0, 28, 1, 100)], 
         "num = ta.highest(close, len) - ta.lowest(close, len)\nden = math.sum(math.abs(ta.change(close)), len)\nvhf = num / (den == 0 ? 1 : den)\nplot(vhf, \"VHF\", color=color.purple)"],
         
        ["FRAMA", "Fractal Adaptive MA", True, [("len", 0, 126, 2, 200), ("fc", 0, 4, 1, 10)], 
         "n3 = (ta.highest(high, len) - ta.lowest(low, len)) / len\nhalf = math.floor(len/2)\nn1 = (ta.highest(high, half) - ta.lowest(low, half)) / (half == 0 ? 1 : half)\nn2 = (ta.highest(high, half)[half] - ta.lowest(low, half)[half]) / (half == 0 ? 1 : half)\nn3_safe = n3 == 0 ? 0.001 : n3\ndimen = (math.log(n1+n2) - math.log(n3_safe)) / math.log(2)\nalpha = math.exp(-4.6 * (dimen - 1))\nvar frama = 0.0\nframa := (alpha * close) + ((1 - alpha) * nz(frama[1], close))\nplot(frama, \"FRAMA\", color=color.blue)"],
         
        ["SMI", "SMI Ergodic", False, [("l", 0, 20, 1, 100), ("s", 0, 5, 1, 50)], 
         "tss(src, l, s) => ta.ema(ta.ema(src, l), s)\nerg = tss(ta.change(close), l, s)\nsig = tss(erg, 5, 5)\nplot(erg, 'Ergodic', color=color.blue)\nplot(sig, 'Signal', color=color.red)"],

        # 51-60
        ["RAVI", "Range Action Verification", False, [("f", 0, 7, 1, 20), ("s", 0, 65, 1, 100)], 
         "ravi = math.abs(ta.sma(close, f) - ta.sma(close, s)) / (ta.sma(close, s) == 0 ? 1 : ta.sma(close, s)) * 100\nplot(ravi, \"RAVI\", color=color.blue)\nhline(3.0)"],
         
        ["STC", "Schaff Trend Cycle", False, [("len", 0, 10, 1, 50), ("f", 0, 23, 1, 100), ("s", 0, 50, 1, 100)], 
         "m = ta.ema(close, f) - ta.ema(close, s)\nstc = ta.sma(m, len)\nplot(stc, 'STC', color=color.blue)\nhline(25)\nhline(75)"],
         
        ["VWAP", "VWAP Bands", True, [("mult", 1, 1.0, 1, 5)], 
         "v = ta.vwap(close)\ns = ta.stdev(close, 20)\nplot(v, 'VWAP', color=color.orange)\nplot(v + s*mult, 'Up')\nplot(v - s*mult, 'Dn')"],
         
        ["COG", "Center of Gravity", False, [("len", 0, 10, 1, 50)], 
         "num = 0.0\nden = 0.0\nfor i = 0 to len-1\n    num += (close[i] * (i + 1))\n    den += close[i]\ncog = -num / (den == 0 ? 1 : den)\nplot(cog, \"COG\", color=color.red)"],
         
        ["ConnorsRSI", "Connors RSI", False, [("r", 0, 3, 1, 20), ("s", 0, 2, 1, 20), ("rank", 0, 100, 1, 200)], 
         "rsi = ta.rsi(close, r)\nvar streak = 0\nstreak := close > close[1] ? nz(streak[1]) + 1 : close < close[1] ? nz(streak[1]) - 1 : 0\nrsistreak = ta.rsi(streak, s)\npr = ta.percentrank(ta.change(close), rank)\ncrsi = (rsi + rsistreak + pr) / 3\nplot(crsi, \"CRSI\", color=color.blue)"],
         
        ["VIDYA", "VIDYA", True, [("len", 0, 9, 1, 50), ("fix", 0, 9, 1, 50)], 
         "cmo = math.abs(ta.cmo(close, fix)) / 100\nvar vidya = 0.0\nalpha = 2 / (len + 1)\nvidya := (alpha * cmo * close) + ((1 - (alpha * cmo)) * nz(vidya[1], close))\nplot(vidya, \"VIDYA\", color=color.blue)"],
         
        ["RVI", "Relative Vigor Index", False, [("len", 0, 10, 1, 50)], 
         "rv = ta.rvi(len)\nsig = ta.swma(rv)\nplot(rv, 'RVI', color=color.green)\nplot(sig, 'Sig', color=color.red)"],
         
        ["Correlation", "Correlation Coefficient", False, [("len", 0, 20, 1, 100), ("idx", 2, "SP:SPX", None, None, "Benchmark")], 
         "bench = request.security(idx, timeframe.period, close[1])\ncor = ta.correlation(close, bench, len)\nplot(cor, \"Corr\", color=color.purple)"],
         
        ["HV", "Historical Volatility", False, [("len", 0, 20, 1, 100)], 
         "hv = ta.stdev(math.log(close / close[1]), len) * math.sqrt(252) * 100\nplot(hv, 'HV', color=color.blue)"],
         
        ["Ulcer", "Ulcer Index", False, [("len", 0, 14, 1, 100)], 
         "h = ta.highest(close, len)\ndd = 100 * (close - h) / (h == 0 ? 1 : h)\nui = math.sqrt(math.sum(dd*dd, len) / len)\nplot(ui, \"UI\", color=color.red)"],

        # 61-70
        ["WCL", "Weighted Close", True, [], 
         "wcl = (high + low + 2 * close) / 4\nplot(wcl, 'WCL', color=color.blue)"],
         
        ["STARC", "STARC Bands", True, [("len", 0, 6, 1, 20), ("mult", 1, 1.3, 1, 5)], 
         "m = ta.sma(close, len)\natr = ta.atr(10)\nplot(m, 'Mid')\nplot(m + atr*mult, 'Up')\nplot(m - atr*mult, 'Dn')"],
         
        ["RWI", "Random Walk Index", False, [("len", 0, 14, 1, 100)], 
         "atr = ta.atr(len)\ndiv = atr * math.sqrt(len)\nrwiH = (high - low[len]) / (div == 0 ? 1 : div)\nrwiL = (high[len] - low) / (div == 0 ? 1 : div)\nplot(rwiH, \"High\", color=color.green)\nplot(rwiL, \"Low\", color=color.red)"],
         
        ["Laguerre", "Laguerre RSI", False, [("g", 1, 0.5, 0, 1)], 
         "var L0 = 0.0\nvar L1 = 0.0\nvar L2 = 0.0\nvar L3 = 0.0\nL0 := (1 - g) * close + g * nz(L0[1])\nL1 := -g * L0 + nz(L0[1]) + g * nz(L1[1])\nL2 := -g * L1 + nz(L1[1]) + g * nz(L2[1])\nL3 := -g * L2 + nz(L2[1]) + g * nz(L3[1])\ncu = (L0 >= L1 ? L0 - L1 : 0) + (L1 >= L2 ? L1 - L2 : 0) + (L2 >= L3 ? L2 - L3 : 0)\ncd = (L0 < L1 ? L1 - L0 : 0) + (L1 < L2 ? L2 - L1 : 0) + (L2 < L3 ? L3 - L2 : 0)\nrsi = cu + cd == 0 ? 0 : cu / (cu + cd)\nplot(rsi, \"Lag RSI\", color=color.blue)"],
         
        ["VixFix", "CM Williams Vix Fix", False, [("len", 0, 22, 1, 100)], 
         "h = ta.highest(close, len)\nvf = (h - low) / (h == 0 ? 1 : h) * 100\nplot(vf, \"Vix Fix\", color=color.red)"],
         
        ["Twiggs", "Twiggs Money Flow", False, [("len", 0, 21, 1, 100)], 
         "trH = math.max(high, close[1])\ntrL = math.min(low, close[1])\ntr = trH - trL\nadv = ((close - trL) - (trH - close)) / (tr == 0 ? 1 : tr) * volume\ntmf = ta.ema(adv, len) / (ta.ema(volume, len) == 0 ? 1 : ta.ema(volume, len)) * 100\nplot(tmf, \"Twiggs\", color=color.green)\nhline(0)"],
         
        ["VQI", "Volatility Quality Index", False, [("len", 0, 20, 1, 100)], 
         "vqi = 0.0\ntr = math.max(high - low, math.abs(high - close[1]), math.abs(low - close[1]))\nvqi := ((close - close[1]) / (tr == 0 ? 1 : tr)) * ((close - open) / (high - low == 0 ? 1 : high - low))\nval = ta.sma(math.abs(vqi) * (close - close[1] + (high - low) / 2), len)\nplot(val, \"VQI\", color=color.purple)"],
         
        ["TII", "Trend Intensity Index", False, [("len", 0, 30, 1, 100)], 
         "ma = ta.sma(close, len)\ndev = close - ma\npos = math.sum(dev > 0 ? dev : 0, len)\nneg = math.sum(dev < 0 ? math.abs(dev) : 0, len)\ntii = 100 * (pos / (pos + neg == 0 ? 1 : pos + neg))\nplot(tii, \"TII\", color=color.blue)\nhline(50)"],
         
        ["QStick", "QStick", False, [("len", 0, 8, 1, 50)], 
         "q = ta.sma(close - open, len)\nplot(q, 'QStick', color=color.blue)\nhline(0)"],
         
        ["Klinger", "Klinger Oscillator", False, [("f", 0, 34, 1, 100), ("s", 0, 55, 1, 100)], 
         "sv = volume * (2 * ((high + low + close) / 3) - (high + low)) / (high + low == 0 ? 1 : high + low)\nko = ta.ema(sv, f) - ta.ema(sv, s)\nplot(ko, \"Klinger\", color=color.blue)"],

        # 71-80
        ["Impulse", "Elder Impulse", False, [("len", 0, 13, 1, 50)], 
         "ema = ta.ema(close, len)\nmacd = ta.ema(close, 12) - ta.ema(close, 26)\nc = (close > ema and macd > macd[1]) ? color.green : (close < ema and macd < macd[1]) ? color.red : color.blue\nplot(1, 'Impulse', color=c, style=plot.style_columns)"],
         
        ["BOP", "Balance of Power", False, [("len", 0, 14, 1, 100)], 
         "bop = (close - open) / (high - low == 0 ? 1 : high - low)\ns = ta.sma(bop, len)\nplot(s, \"BOP\", color=color.blue)\nhline(0)"],
         
        ["Psych", "Psychological Line", False, [("len", 0, 12, 1, 50)], 
         "up = math.sum(close > close[1] ? 1 : 0, len)\npsy = (up / len) * 100\nplot(psy, 'Psych', color=color.purple)\nhline(50)"],
         
        ["VFI", "Volume Flow Indicator", False, [("len", 0, 130, 1, 200)], 
         "coef = 0.2\nvcoef = 2.5\ntyp = hl2\ninter = math.log(typ) - math.log(typ[1])\nvinter = ta.stdev(inter, 30)\ncut = coef * vinter * close\nvave = ta.sma(volume, len)\nvmax = vave * vcoef\nvc = volume < vmax ? volume : vmax\nmf = typ - typ[1] > cut ? vc : typ - typ[1] < -cut ? -vc : 0\nvfi = ta.sma(mf, len) / (vave == 0 ? 1 : vave)\nplot(vfi, \"VFI\", color=color.orange)\nhline(0)"],
         
        ["RMI", "Relative Momentum Index", False, [("len", 0, 20, 1, 100), ("mom", 0, 5, 1, 20)], 
         "up = math.sum(close > close[mom] ? close - close[mom] : 0, len)\ndn = math.sum(close < close[mom] ? close[mom] - close : 0, len)\nrmi = 100 * up / (up + dn)\nplot(rmi, 'RMI', color=color.blue)\nhline(70)\nhline(30)"],
         
        ["REI", "Range Expansion Index", False, [("len", 0, 8, 1, 50)], 
         "h2 = high[2]\nl2 = low[2]\ndiff = high - low\ncond = (high >= l2 and high <= h2) or (low >= l2 and low <= h2)\nval = cond ? 0 : (high - h2) + (low - l2)\nrei = math.sum(val, len) / (math.sum(diff, len) == 0 ? 1 : math.sum(diff, len)) * 100\nplot(rei, \"REI\", color=color.blue)\nhline(60)\nhline(-60)"],
         
        ["ProjOsc", "Projection Oscillator", False, [("len", 0, 14, 1, 100)], 
         "slope = ta.linreg(close, len, 0) - ta.linreg(close, len, 1)\nu = ta.highest(slope, len)\nl = ta.lowest(slope, len)\npo = 100 * (slope - l) / (u - l == 0 ? 1 : u - l)\nplot(po, \"ProjOsc\", color=color.purple)\nhline(80)\nhline(20)"],
         
        ["IMI", "Intraday Momentum Index", False, [("len", 0, 14, 1, 100)], 
         "up = math.sum(close > open ? close - open : 0, len)\ndn = math.sum(close < open ? open - close : 0, len)\nimi = 100 * up / (up + dn)\nplot(imi, 'IMI', color=color.blue)\nhline(70)\nhline(30)"],
         
        ["NVI", "Negative Volume Index", False, [], 
         "var nvi = 1000.0\nif volume < volume[1]\n    nvi := nvi[1] + ((close - close[1]) / (close[1] == 0 ? 1 : close[1]) * nvi[1])\nplot(nvi, 'NVI', color=color.red)"],
         
        ["PVI", "Positive Volume Index", False, [], 
         "var pvi = 1000.0\nif volume > volume[1]\n    pvi := pvi[1] + ((close - close[1]) / (close[1] == 0 ? 1 : close[1]) * pvi[1])\nplot(pvi, 'PVI', color=color.green)"],

        # 81-90
        ["OBV_Osc", "OBV Oscillator", False, [("len", 0, 20, 1, 100)], 
         "obv = ta.obv()\nosc = obv - ta.sma(obv, len)\nplot(osc, \"OBV Osc\", color=color.blue)\nhline(0)"],
         
        ["Slope", "Linear Regression Slope", False, [("len", 0, 14, 1, 100)], 
         "s = ta.linreg(close, len, 0) - ta.linreg(close, len, 1)\nplot(s, 'Slope', color=color.orange)\nhline(0)"],
         
        ["Inertia", "Inertia", False, [("len", 0, 20, 1, 100)], 
         "rvi = ta.rvi(14)\ninertia = ta.linreg(rvi, len, 0)\nplot(inertia, 'Inertia', color=color.blue)\nhline(50)"],
         
        ["RVI_Vol", "Relative Volatility Index", False, [("len", 0, 14, 1, 100)], 
         "st = ta.stdev(close, 10)\nu = math.sum(close > close[1] ? st : 0, len)\nd = math.sum(close < close[1] ? st : 0, len)\nrvi = 100 * u / (u + d)\nplot(rvi, 'RVI Vol', color=color.purple)\nhline(50)"],
         
        ["VolRatio", "Volatility Ratio", False, [("len", 0, 14, 1, 100)], 
         "tr = ta.tr()\nvr = tr / (ta.ema(tr, len) == 0 ? 1 : ta.ema(tr, len))\nplot(vr, \"VR\", color=color.red)\nhline(0.5)"],
         
        ["PGO", "Pretty Good Oscillator", False, [("len", 0, 14, 1, 100)], 
         "pgo = (close - ta.sma(close, len)) / (ta.atr(len) == 0 ? 1 : ta.atr(len))\nplot(pgo, \"PGO\", color=color.blue)\nhline(2.0)"],
         
        ["Swing", "Swing Index", False, [], 
         "k = math.max(math.abs(high - close[1]), math.abs(low - close[1]))\nsi = 50 * ((close - close[1]) + 0.5 * (close - open) + 0.25 * (close[1] - open[1])) / (k == 0 ? 1 : k)\nplot(si, 'SI', color=color.green)"],
         
        ["ASI", "Accumulation Swing Index", False, [], 
         "k = math.max(math.abs(high - close[1]), math.abs(low - close[1]))\nsi = 50 * ((close - close[1]) + 0.5 * (close - open) + 0.25 * (close[1] - open[1])) / (k == 0 ? 1 : k)\nvar asi = 0.0\nasi := nz(asi[1]) + si\nplot(asi, 'ASI', color=color.purple)"],
         
        ["PriceOsc", "Price Oscillator", False, [("s", 0, 12, 1, 50), ("l", 0, 26, 1, 100)], 
         "po = ta.ema(close, s) - ta.ema(close, l)\nplot(po, 'PO', color=color.blue)\nhline(0)"],
         
        ["Demand", "Demand Index", False, [("len", 0, 19, 1, 100)], 
         "float bp = volume\nfloat sp = volume\nif close > close[1]\n    sp := 0.0\nelse\n    bp := 0.0\ndi = ta.sma(bp, len) / (ta.sma(sp, len) == 0 ? 1 : ta.sma(sp, len))\nplot(di, \"Demand\", color=color.blue)"],

        # 91-100
        ["VolMACD", "Volume Weighted MACD", False, [("f", 0, 12, 1, 50), ("s", 0, 26, 1, 100)], 
         "vf = ta.ema(close * volume, f) / (ta.ema(volume, f) == 0 ? 1 : ta.ema(volume, f))\nvs = ta.ema(close * volume, s) / (ta.ema(volume, s) == 0 ? 1 : ta.ema(volume, s))\nvm = vf - vs\nplot(vm, \"VolMACD\", color=color.green)\nhline(0)"],
         
        ["VolROC", "Volume Rate of Change", False, [("len", 0, 12, 1, 100)], 
         "vroc = ta.roc(volume, len)\nplot(vroc, 'VROC', color=color.blue)\nhline(0)"],
         
        ["AdjCoppock", "Adjusted Coppock", False, [("len", 0, 14, 1, 50)], 
         "c = ta.wma(ta.roc(close, len), len)\nplot(c, 'AdjCoppock', color=color.purple)\nhline(0)"],
         
        ["ALMA", "ALMA", True, [("len", 0, 9, 1, 100), ("sig", 1, 6.0, 1, 10)], 
         "a = ta.alma(close, len, 0.85, sig)\nplot(a, 'ALMA', color=color.blue)"],
         
        ["Median", "Median Price", True, [("len", 0, 20, 1, 100)], 
         "m = (high + low) / 2\nma = ta.sma(m, len)\nplot(ma, 'Median MA', color=color.orange)"],
         
        ["Typical", "Typical Price", True, [("len", 0, 20, 1, 100)], 
         "t = hlc3\nma = ta.sma(t, len)\nplot(ma, 'Typical MA', color=color.blue)"],
         
        ["WMA_Env", "WMA Envelope", True, [("len", 0, 20, 1, 100), ("pct", 1, 2.0, 1, 10)], 
         "w = ta.wma(close, len)\nu = w * (1 + pct/100)\nl = w * (1 - pct/100)\nplot(w, color=color.gray)\nplot(u, color=color.blue)\nplot(l, color=color.blue)"],
         
        ["Ribbon", "EMA Ribbon", True, [("f", 0, 20, 1, 50), ("s", 0, 50, 1, 100)], 
         "e1 = ta.ema(close, f)\ne2 = ta.ema(close, s)\nplot(e1, 'Fast', color=color.green)\nplot(e2, 'Slow', color=color.red)"],
         
        ["ZLEMA", "Zero Lag EMA", True, [("len", 0, 20, 1, 100)], 
         "lag = math.round((len - 1) / 2)\ndata = close + (close - close[lag])\nz = ta.ema(data, len)\nplot(z, 'ZLEMA', color=color.blue)"],
         
        ["TEMA", "Triple EMA", True, [("len", 0, 20, 1, 100)], 
         "ema1 = ta.ema(close, len)\nema2 = ta.ema(ema1, len)\nema3 = ta.ema(ema2, len)\nt = 3 * ema1 - 3 * ema2 + ema3\nplot(t, 'TEMA', color=color.purple)"]
    ]

    for item in indicators:
        filename, title, overlay, params, logic = item
        
        # Build header
        content = [
            "//@version=6",
            f'indicator("{title}", overlay={str(overlay).lower()})',
            ""
        ]
        
        # Build inputs
        if params:
            content.append("// --- Inputs ---")
            for param in params:
                if len(param) == 6:
                    p_name, p_type, p_def, p_min, p_max, p_title = param
                else:
                    p_name, p_type, p_def, p_min, p_max = param
                    p_title = p_name.capitalize()

                if p_type == 0:  # Int
                    content.append(f'int {p_name} = input.int({p_def}, "{p_title}", minval={p_min}, maxval={p_max}, step=1)')
                elif p_type == 1:  # Float
                    content.append(f'float {p_name} = input.float({p_def}, "{p_title}", minval={p_min}, maxval={p_max}, step=1)')
                elif p_type == 2:  # Symbol
                    content.append(f'string {p_name} = input.symbol("{p_def}", "{p_title}")')
                else:
                    raise ValueError(f"Unknown input type for {p_name}: {p_type}")
            content.append("")
            
        # Build logic - replace single quotes with double quotes for Pine Script v6
        logic_fixed = logic.replace("'", "\"")
        content.append("// --- Logic ---")
        content.append(logic_fixed)
        
        # Write file
        filepath = os.path.join(output_dir, f"{filename}.pine")
        with open(filepath, "w") as f:
            f.write("\n".join(content))
            
    print(f"Successfully generated {len(indicators)} Pine Script v6 files in '{output_dir}'.")

if __name__ == "__main__":
    generate_pine_scripts()