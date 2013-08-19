package hex.gbm;

import hex.gbm.DTree.*;
import java.util.Arrays;
import java.util.Random;
import water.*;
import water.H2O.H2OCountedCompleter;
import water.fvec.*;
import water.util.Log.Tag.Sys;
import water.util.Log;
import water.util.Utils;

// Random Forest Trees
public class DRF extends Job {
  public static final String KEY_PREFIX = "__DRFModel_";

  // The Key of the Forest
  public Key _forestKey;

  long _cm[/*actual*/][/*predicted*/]; // Confusion matrix
  public long[][] cm() { return _cm; }

  public static final Key makeKey() { return Key.make(KEY_PREFIX + Key.make());  }
  private DRF(Key dest, Frame fr) { super("DRF "+fr, dest); }
  // Called from a non-FJ thread; makea a DRF and hands it over to FJ threads
  public static DRF start(Key dest, final Frame fr, final int maxDepth, final int ntrees, final int mtrys, final double sampleRate, final long seed) {
    final DRF job = new DRF(dest, fr);
    H2O.submitTask(job.start(new H2OCountedCompleter() {
        @Override public void compute2() { job.run(fr,maxDepth,ntrees,mtrys,(float)sampleRate,seed); tryComplete(); }
      }));
    return job;
  }

  // ==========================================================================

  // Compute a DRF forest.
  //
  // This method just builds a set of deterministic tree-seeds, then hands out
  // jobs.  Nodes pull tree-seeds as they are free, build one tree, then go
  // back for another tree.

  // Compute a single DRF tree from the Frame.  Last column is the response
  // variable.  Depth is capped at maxDepth.
  private void run(final Frame fr, final int maxDepth, final int ntrees, final int mtrys, final float sampleRate, long seed ) {
    Timer t_drf = new Timer();
    assert 0 <= ntrees && ntrees < 1000000;
    assert 0 <= mtrys && mtrys < fr.numCols();
    assert 0.0 < sampleRate && sampleRate <= 1.0;

    final String names[] = fr._names;
    Vec vs[] = fr._vecs;
    final int ncols = vs.length-1; // Last column is the response column

    // Response column is the last one in the frame
    Vec vresponse = vs[ncols];
    final long nrows = vresponse.length();
    assert !vresponse._isInt || (vresponse.max() - vresponse.min()) < 10000; // Too many classes?
    int ymin = (int)vresponse.min();
    final short nclass = vresponse._isInt ? (short)(vresponse.max()-ymin+1) : 0;

    // Make a master list of trees to be worked on.
    // Make a deterministic set of RNG seeds used to pick split columns.
    final Forest forest = new Forest(ncols, ymin, nclass, ntrees,Utils.getRNG(seed));
    _forestKey = Key.make(Key.make().toString(),(byte)0,Key.BUILT_IN_KEY);
    UKV.put(_forestKey,forest);

    // Get Nodes busy making trees.
    new DRemoteTask() {
      @Override public void lcompute() {
        int tnum;
        while( (tnum = ((GetOne)new GetOne().invoke(_forestKey))._tnum) != -1 )
          doTree(_forestKey,fr,nclass,maxDepth,mtrys,sampleRate,forest._seeds,tnum);
        tryComplete();
      }
      @Override public void reduce(DRemoteTask drt) { }
    }.invokeOnAllNodes();

    // Get the updated list
    System.out.println("All trees done! "+t_drf);

    Forest forest2 = UKV.get(_forestKey);
    // Compute a OOBEE Confusion Matrix
    _cm = new OOBEETask(forest2,sampleRate,fr).invokeOnAllNodes()._cm;
    System.out.println("Confusion Matrix ");
    for( long ls[] : _cm )
      System.out.println(Arrays.toString(ls));

    System.out.println(forest2);
  }

  // Master list of tree-seeds, and which trees are being worked on.
  // This list is atomically updated by nodes as they work on trees.
  public static class Forest extends Iced/*should be an H2O Model class*/ {
    final int _ncols;           // Number of columns in the model
    final int _ymin;            // Smallest class used to model with
    final int _nclass;          // Number of classes from ymin to ymin+_nclass-1
    final long [] _seeds;       // Random gen seeds per-tree
    final Key [] _treeKeys;     // Tree Keys, set as trees are completed
    int _next;                  // Next un-owned tree-gen
    Forest( int ncols, int ymin, int nclass, int ntrees, Random rand ) {
      _ncols = ncols;  _ymin = ymin;  _nclass = nclass;
      _treeKeys = new Key[ntrees];
      _seeds = new long[ntrees];
      for( int i=0; i<ntrees; i++ )
        _seeds[i] = rand.nextLong();
    }
    public void deleteKeys() {
      Futures fs = new Futures();
      for( Key k : _treeKeys ) UKV.remove(k,fs);
      fs.blockForPending();
    }

    // Compute Model stats; tree size, depth, nodes
    transient long _t_bytes, _t_depth, _t_nodes;
    public long t_bytes() { return _t_bytes == 0 ? modelStats()._t_bytes : _t_bytes; }
    public long t_depth() { return _t_depth == 0 ? modelStats()._t_depth : _t_depth; }
    public long t_nodes() { return _t_nodes == 0 ? modelStats()._t_nodes : _t_nodes; }
    private Forest modelStats() {
      for( int i=0; i<_treeKeys.length; i++ ) {
        Key k = _treeKeys[i];
        byte[] bits = DKV.get(k).getBytes();
        assert UDP.get4(bits,0/*tree id*/)==i; // Check tree-bits for sane ID's
        long d_n = DRFTree.stats(bits);
        int depth = (int)(d_n>>32);
        int nodes = (int)(d_n    );
        _t_bytes += bits.length;
        _t_depth += depth;
        _t_nodes += nodes;
      }
      return this;
    }

    @Override public String toString() {
      return "Trees#"+_treeKeys.length+
        ", avg bytes="+(t_bytes()/_treeKeys.length)+
        ", avg nodes="+(t_nodes()/_treeKeys.length)+
        ", max depth="+(t_depth()/_treeKeys.length);
    }
  }

  // Atomically fetch a tree# to work on & claim it, or return -1.
  private static class GetOne extends TAtomic<Forest> {
    int _tnum = -1;
    @Override public Forest atomic( Forest old ) {
      if( old._next >= old._seeds.length ) return null;
      _tnum = old._next++;
      return old;
    }
  }

  private static class SetOne extends TAtomic<Forest> {
    final int _tnum;
    Key _treeKey;
    SetOne( int tnum, Key treeKey ) { _tnum = tnum; _treeKey = treeKey; }
    @Override public Forest atomic( Forest old ) {
      assert old._treeKeys[_tnum] == null : "Need to ignore & free extra dup tree";
      old._treeKeys[_tnum] = _treeKey;
      return old;
    }
  }

  // --------------------------------------------------------------------------
  // Compute a single RF tree
  private static void doTree( Key forestKey, Frame fr, short nclass, int maxDepth, int mtrys, float sampleRate, long seeds[], int tnum ) {
    Timer t_tree = new Timer();
    System.out.println("Working on tree "+tnum);

    // Ask all Nodes to setup for building this one tree, and hand me back a
    // cookie (Key) to guide further tree building.
    DRFTree root = new Start1Tree(fr).invokeOnAllNodes()._drf;
    // Capture and reset root info
    final int[] ends = root._begs;
    root._begs = new int[ends.length];
    Key keys[] = root._keys;

    // Split the top-level tree based on random sampling.
    // root._kids[0] will be sampled-in, and root._kids[1] will be sampled-out
    final long seed = seeds[tnum];
    new SampleSplit(keys,root._begs,ends,sampleRate,seed).invokeOnAllNodes().makeKids(root);    
    DRFTree train = root._kids[0];
    DRFTree test  = root._kids[1];
    train._keys = keys;         // Put these back (not passed back)
    // End of tree 'train' is also the beginning of tree 'test'.
    final int[] ends0 = test._begs;

    // Build an initial set of columns to begin tree-splitting on
    int ncols=fr.numCols()-1;   // Last column is response
    int   []cols = new int  [ncols];
    float []mins = new float[ncols];
    float []maxs = new float[ncols];
    int j=0;
    for( int i=0; i<ncols; i++ ) {
      Vec v = fr._vecs[i];
      cols[j] = i;
      mins[j] = (float)v.min();
      maxs[j] = (float)v.max();
      if( mins[i] != maxs[i] ) j++; // Drop out constant columns
    }
    if( j<ncols ) {             // Shrink arrays if any columns dropped
      cols = Arrays.copyOf(cols,j);
      mins = Arrays.copyOf(mins,j);
      maxs = Arrays.copyOf(maxs,j);
    }

    // Build a RNG from the seed.
    long seed2 = seed+ends0[0];
    Random rand = Utils.getRNG(seed2);

    // build a tree
    train.doSplit(fr,ends0,cols,mins,maxs,rand,mtrys,0,maxDepth,nclass);
    //train.print(fr._names);

    // Compact the main tree to make a dense model.
    byte[] bits = root.compact(new AutoBuffer(),tnum,seed).buf();
    // Install it in the Forest
    Key tkey = Key.make("treeKey"+Key.make(),(byte)0,Key.BUILT_IN_KEY);
    UKV.put(tkey,new Value(tkey,bits));
    new SetOne(tnum,tkey).invoke(forestKey);

    // Remove local temp tree storage
    Futures fs = new Futures();
    for( Key k : keys ) UKV.remove(k,fs);
    fs.blockForPending();

    System.out.println("Done on tree "+tnum+", size="+bits.length+" in "+t_tree);
  }

  // The simple tree-node class.  Every tree node has a list of local-rows that
  // landed on that tree node; the list is large and kept in the LocalTree
  // structure on each JVM.  We only keep the start & end region marker in the
  // large LocalTree here.
  private static class DRFTree extends Iced {
    // INPUTS, cleared when done
    Key _keys[/*nodes*/];    // Keys for local data for this one tree
    // INPUT/OUTPUT - output/set on 1st pass, input thereafter
    int _begs[];             // Beginning-row for this tree leaf, for all nodes
    // OUTPUT - set after splitting decisions
    int _col;                // Column to split on
    float _min, _step;       // lower-bound & step size for which kid
    DRFTree _kids[];         // Kids, if any.
    int _clss[/*nclass*/];   // Class distribution at leaves, or null in the interior
    transient int _byteSize; // Compacted size

    // Called to make a Root DRFTree.  Passed in a key, and the length of the
    // entire row list.  Sets the Key list.
    DRFTree( Key key, int end ) {
      _keys = new Key[H2O.CLOUD.size()];
      _begs = new int[_keys.length];
      int idx = H2O.SELF.index();
      _keys[idx] = key;
      _begs[idx] = end;
    }
    // Merge trees on the Root DRFTree
    void add1( DRFTree drf ) {
      assert _kids==null;
      for( int i=0; i<_keys.length; i++ )
        if( drf._keys[i] != null ) {
          _keys[i] = drf._keys[i];
          _begs[i] = drf._begs[i];
        }
    }

    // Called to make a new interior DRFTree.  Just the row starts.
    DRFTree( int begs[] ) { _begs = begs; }

    // ---
    // Build a tree level.  Passed in a set of active columns, and their min/max's.
    // Conceptually, repeat for mtrys columns:
    // (0) Figure a column to split on, drawn at random without replacement.
    // (1) build histograms of those columns
    // (2) figure out best split
    // (3) split rows
    // Then (4) recursively compute a new level by doing the above steps on
    // each of the several splits.
    void doSplit(Frame fr, int ends[/*nodes*/], int cols[/*cols*/], float mins[/*cols*/], float maxs[/*cols*/], Random rand, int mtrys, int depth, int maxDepth, short nclass) {
      if( depth >= maxDepth ) return;

      // We pick up to mtrys columns, pulling at random without replacement
      // from the entire list of columns.  Picked columns are moved to the end
      // of the column list, and are not picked again.
      int res_col=cols.length;
      DHisto2 best_hist = null;
      float best_score = Float.MAX_VALUE;
      for( int i=0; i<mtrys; i++ ) {
        if( res_col == 0 ) break; // Out of choices!
        // (0) Figure a column to split on, drawn at random without replacement.
        int cidx = rand.nextInt(res_col);

        // (1) build histograms & score
        int col = cols[cidx];
        DHisto2 dh2 = new DHisto2(this,ends,nclass,col,mins[cidx],maxs[cidx],fr._vecs[col]._isInt,fr._names[col]);
        float f = dh2.invokeOnAllNodes().scoreClassification();
        // Tighten mins & maxs based on what is discovered
        float min = dh2.min();  if( min > mins[cidx] ) mins[cidx] = min;
        float max = dh2.max();  if( max < maxs[cidx] ) maxs[cidx] = max;

        // (2) figure out best split
        if( f < best_score ) { best_score = f;  best_hist = dh2; }
        if( f == 0 ) break;     // No error?  Cannot improve; stop now

        // Move selected column to the end of res_col, so we do not pick it again
        swap(cols,cidx,--res_col);
        swap(mins,cidx,  res_col);
        swap(maxs,cidx,  res_col);

        // See if this column is useless to predict with.  Remove useless columns.
        if( min == max ) {
          assert cols[res_col] == col;
          int last = cols.length-1;
          swap(cols,res_col,last);  cols = Arrays.copyOf(cols,last);
          swap(mins,res_col,last);  mins = Arrays.copyOf(mins,last);
          swap(maxs,res_col,last);  maxs = Arrays.copyOf(maxs,last);
          i--;                  // Also, pick another column
        }
      }
      assert best_hist != null : indent(depth)+"Out of columns to split on, "+this.toString(ends);

      // A constant response means this is a perfectly predicting leaf
      if( best_hist.constantResponse() ) {
        _clss = new int[nclass]; // Record the class distribution
        for( int b=0; b<best_hist._nbins; b++ )
          for( int c =0; c<nclass; c++ )
            _clss[c] += best_hist._clss[b][c];
        return;
      }

      // Record enough info to 'score' a row on this node
      _col = best_hist._col;
      _min = best_hist._min ;
      _step= best_hist._step;
      //System.out.print(indent(depth)+fr._names[_col]+", choices: ");
      //for( int c=0; c<cols.length; c++ )
      //  System.out.print(fr._names[cols[c]]+"/"+mins[c]+"/"+maxs[c]+" ");
      //System.out.println();
      //for( int i=0; i<best_hist._clss.length; i++ )
      //  System.out.println(indent(depth)+Arrays.toString(best_hist._clss[i]));

      int cidx = cols.length-1; // Find the column in my list of active ones; might be removed
      while( cidx >= 0 && cols[cidx] != best_hist._col ) cidx--;
      assert cidx== -1 || cols[cidx]==_col;

      // (3) split rows based on best histogram/column
      Key keys[] = _keys;       // Save for the kids
      new HistoSplit(_keys,_begs,ends,best_hist).invokeOnAllNodes().makeKids(this);

      // (4) recursively compute new levels
      int nkids = best_hist._nbins;
      for( int k=0; k<nkids; k++ ) {
        int local_ends[] = (k+1 < nkids) ?_kids[k+1]._begs : ends;
        // Skip empty bins
        int nelems=0;             // Count elements across the cluster in this split
        for( int i=0; i<local_ends.length; i++ )
          nelems += local_ends[i]-_kids[k]._begs[i];
        if( nelems == 0 ) continue; // Bin is empty?  This tree ends here

        // Make private copys of these arrays; each subtree proceeds in
        // parallel and tightens these arrays as it goes.
        int   cols2[] = cols.clone(); // All remaining columns to work on
        float mins2[] = mins.clone(); // Their mins & maxes
        float maxs2[] = maxs.clone(); // Their mins & maxes
        // For the chosen column, for each subsplit, we can make tighter min/max
        if( cidx != -1 ) {      // Constant column might already be removed
          mins2[cidx] = best_hist._mins[k];
          maxs2[cidx] = best_hist._maxs[k];
          // If we shrink to a constant, remove it: cannot split a constant
          if( mins2[cidx] == maxs2[cidx] ) {
            int last = cols.length-1;
            swap(cols2,cidx,last);  cols2 = Arrays.copyOf(cols2,last);
            swap(mins2,cidx,last);  mins2 = Arrays.copyOf(mins2,last);
            swap(maxs2,cidx,last);  maxs2 = Arrays.copyOf(maxs2,last);
          }
        }
        // Any sane columns to split on?
        if( cols2.length == 0 ) { // No more columns to split, make a leaf
          _kids[k]._clss = new int[nclass];
          for( int c=0; c<nclass; c++ )
            _kids[k]._clss[c] = best_hist._clss[k][c];
        } else {                // Split this set of rows
          Random rand2 = Utils.getRNG(rand.nextLong());
          _kids[k]._keys = keys;
          _kids[k].doSplit(fr,local_ends,cols2,mins2,maxs2,rand2,mtrys,depth+1,maxDepth,nclass);
        }
      }
    }

    // ---
    // Quick sanity check on bounds
    boolean check( int[] ends ) {
      for( int i=0; i<ends.length; i++ )
        if( _begs[i] > ends[i] )
          return false;
      return true;
    }

    // Pretty-print
    @Override public String toString() {
      String s = "{"+_col+"/"+_min+"/"+_step+",";
      int sz = H2O.CLOUD.size();
      for( int i=0; i<sz; i++ )
        s += "["+H2O.CLOUD._memary[i]+","+(_begs==null ? -1 : _begs[i])+"],";
      return s+"}";
    }
    // Pretty-print with the other ends
    public String toString( int[] ends ) {
      String s = "{";
      for( int i=0; i<_begs.length; i++ )
        s += "["+H2O.CLOUD._memary[i]+","+_begs[i]+","+ends[i]+"],";
      return s+"}";
    }

    public static String indent( int d ) {
      String s="";
      for( int i = 0; i<d; i++ ) s += "  ";
      return s;
    }

    // Pretty-print a tree.  Warning: trees can get large.  You get 1 line per
    // node in the tree.
    public void print(String names[]) { System.out.println(print(new StringBuilder(),names,0)); }
    private StringBuilder print(StringBuilder sb, String[]names, int d) {
      for( int i = 0; i<d; i++ ) sb.append("  "); // Indent
      if( _clss == null ) {     // Interior?
        if( _kids == null ) {   // No class, no kids?
          sb.append("null\n");  // No rows to train on, no prediction.
        } else {
          sb.append("{col=").append(names[_col]);
          float f = _min;
          for( int i=0; i<_kids.length; i++,f+=_step ) sb.append(',').append(f);
          sb.append("}\n");
          for( int i=0; i<_kids.length; i++ ) _kids[i].print(sb,names,d+1);
        }
      } else {                  // Leaf?
        assert _kids == null;
        sb.append("{class= ");
        for( int i=0; i<_clss.length; i++ ) sb.append(_clss[i]).append(",");
        sb.append("}\n");
      }
      return sb;
    }

    // Build the compressed form of a tree.
    //  +1b #nbins,1-254.  Has child splits.  +2b column to split
    //     For each bin 2-till last: +4b float bin max, {+1,+4} skip, then subtree bytes
    //     Last bin: just subtree bytes
    //  +1b 0 - Has class distribution in bytes
    //     +1b * nclasses, 1 byte per class
    //  +1b 255 - Has class distribution in ints
    //     +4b * nclasses, 1 int per class
    public AutoBuffer compact( AutoBuffer ab, int id, long seed ) {
      return _kids[0].compact(ab.put4(id).put8(seed));
    }
    private AutoBuffer compact( AutoBuffer ab ) {
      if( _clss == null ) {     // Interior?
        assert 0 <= _col && _col <= 65535;
        assert _kids != null;   // Already handled before calling
        if( _kids.length > 254 ) throw H2O.unimpl();  // Need another format for giant splits?
        int nbins = 0;
        for( int i=0; i<_kids.length; i++ )
          if( !_kids[i].noPrediction() ) nbins++; // Count useful splits
        assert nbins >= 1;       // Must be splitting something
        int last=_kids.length-1; // Last bin with a prediction
        while( _kids[last].noPrediction() ) last--;
        ab.put1(nbins).put2((char)_col); // Bin count & column to split on
        float f = _min;
        for( int i=0; i<_kids.length; i++ ) {
          f += _step;                    // Get bin-max
          if( _kids[i].noPrediction() ) continue; // Skip over a no-row-no-prediction child
          if( i<last ) {                // Last bin just has subtree bytes
            ab.put4f(f);                // Value for bin-max
            int skip = _kids[i].size(); // Drop down the amount to skip over
            if( skip <= 254 ) ab.put1(skip);
            else ab.put1(0).put3(skip);
          }
          _kids[i].compact(ab); // Subtree bytes
        }
      } else {                  // Leaf?
        long max=Long.MIN_VALUE;
        for( int c : _clss ) max=Math.max(max,c);
        if( max < 256 ) { ab.put1(  0); for( int c : _clss ) ab.put1((byte)c); }
        else            { ab.put1(255); for( int c : _clss ) ab.put4(      c); }
      }
      return ab;
    }

    // Compacted byteSize.  Structurally matches the compact() function above,
    // but does an abstract length computation instead the full bits.
    final int size( ) { return _byteSize==0 ? (_byteSize=size_impl()) : _byteSize; }
    private int size_impl( ) {
      if( _clss == null ) {
        assert _kids != null;   // Already handled before calling
        int nbins = 0;
        for( int i=0; i<_kids.length; i++ )
          if( !_kids[i].noPrediction() ) nbins++; // Count useful splits
        assert nbins >= 1 : "nbin="+nbins+", col="+_col+", fmin="+_min;      // Must be splitting something
        int last=_kids.length-1;                 // Last bin with a prediction
        while( _kids[last].noPrediction() ) last--;
        int sz=1+2;             // nbins & column
        for( int i=0; i<_kids.length; i++ ) {
          if( _kids[i].noPrediction() ) continue;
          int szk = _kids[i].size();
          if( i<last )
            sz += 4/*Float bin-max*/ + (szk <= 254 ? 1 : 4)/*skip size*/;
          sz += szk; // Subtree bytes
        }
        return sz;
      } else {
        long max=Long.MIN_VALUE;
        for( int c : _clss ) max=Math.max(max,c);
        if( max < 256 ) return 1+1*_clss.length; // Class distribution as 1 byte
        else            return 1+4*_clss.length; // Class distribution as 4 byte
      }
    }

    private boolean noPrediction() { return _clss==null && _kids==null; }

    // Score a row on a pile-o-bits
    public static void score( byte bits[], Frame fr, long row, int votes[] ) {
      int idx = 4/*tnum*/+8/*seed*/;
      Vec vecs[] = fr._vecs;
      OUTER:
      while( true ) {
        int nbin = bits[idx++]&0xFF;
        if( nbin == 0 ) {       // byte-wise class distribution
          for( int i=0; i<votes.length; i++ )
            votes[i] += bits[idx++]&0xFF; // Add in votes
          return;
        } else if( nbin == 255 ) {
          for( int i=0; i<votes.length; i++ )
            { votes[i] += UDP.get4(bits,idx); idx += 4; } // Add in votes
          return;
        } else {
          int col = UDP.get2(bits,idx); idx+=2;
          float f = (float)(vecs[col].at(row));
          while( nbin > 1 ) {
            float fx = UDP.get4f(bits,idx); idx += 4;
            int skip = bits[idx++]&0xFF;
            if( skip == 0 ) { skip = UDP.get3(bits,idx); idx+=3; }
            if( f < fx ) continue OUTER; // Continue on outer loop in this subtree
            idx += skip;                 // Else skip subtree
            nbin--;
          }
          // Continue on outer loop in final subtree
        }
      }
    }

    private static long merge_d_n(int depth, int nodes ) { return ((long)depth)<<32 | nodes; }
    public static long stats( byte bits[] ) { return stats(bits,4/*tnum*/+8/*seed*/,0,0,0); }
    private static long stats( byte bits[], int idx, int depth, int maxd, int nodes ) {
      nodes++;
      if( depth > maxd ) maxd = depth;
      int nbin = bits[idx++]&0xFF;
      if( nbin == 0 || nbin==255 )
        return merge_d_n(maxd,nodes);
      idx += 2;                 // Column
      while( nbin > 1 ) {
        idx += 4;               // Float
        int skip = bits[idx++]&0xFF;
        if( skip == 0 ) { skip = UDP.get3(bits,idx); idx+=3; }
        long d_n = stats(bits,idx,depth+1,maxd,nodes);
        maxd  = (int)(d_n>>32);
        nodes = (int)(d_n    );
        idx += skip;
        nbin--;
      }
      // Last guy has no float/skip bytes
      return stats(bits,idx,depth+1,maxd,nodes);
    }
  }

  // ---
  // Start this one Tree on all Nodes.  Return a collection of Keys that
  // uniquely identifies the tree being built on all Nodes.
  private static class Start1Tree extends DRemoteTask<Start1Tree> {
    final Frame _fr;            // INPUT frame
    DRFTree _drf;               // OUTPUT result top-level tree
    Start1Tree(Frame fr) { _fr = fr; }
    @Override public void lcompute() {
      // One Key per Tree, homed to self.
      Key myKey = Key.make("T"+Key.make(),(byte)0,Key.BUILT_IN_KEY,H2O.SELF);
      // Stash away local tree info on this Node, awaiting the next pass.
      LocalTree lt = new LocalTree(myKey,_fr);
      UKV.put(myKey,lt);
      _drf = new DRFTree(myKey,lt._rows0.length);
      tryComplete();
    }
    // Combine Key/beg lists from across the cluster
    @Override public void reduce(Start1Tree s1t) {
      if( s1t == null ) return;
      if( _drf == null ) _drf = s1t._drf;
      else _drf.add1(s1t._drf);
    }
  }

  // ---
  // Generic "split a tree leaf" based on some function.  The split amounts to
  // having each Node segregate the data rows from 'beg' to 'end' into 2
  // sections, one from 'beg' to the midpoint, and the other from midpoint to
  // the end.  The actual rows are kept in the giant LocalTree array on each Node.
  private static abstract class Split<T extends Split> extends DRemoteTask<T> {
    // INPUT - cleared after local work
    Key _keys[/*nodes*/];    // Keys for local data for this one tree
    // INPUT/OUTPUT - output/set on 1st pass, input thereafter
    int _begs[/*nodes*/];    // Beginning-row for this tree leaf, for all nodes
    int _ends[/*nodes*/];    // Ends of split rows
    // OUTPUT
    int _kids[/*nkids*/][/*nodes*/];
    // Subclass provided function describing the split decision
    abstract void doSplit(LocalTree lt, int beg, int end);

    Split( Key keys[], int begs[], int[] ends ) {
      _keys = keys;
      _begs = begs;
      _ends = ends;
    }
    @Override public void lcompute() {
      int idx = H2O.SELF.index();
      LocalTree lt = DKV.get(_keys[idx]).get();
      int beg = _begs[idx];
      int end = _ends[idx];
      // Ordered before the split, so the split can do efficient visitation
      assert lt.isOrdered(beg,end) : "Not ordered";

      doSplit(lt,beg,end);

      // Children (if any) are ordered after the split, so the next split layer
      // starts ordered
      if( _kids != null ) {
        int x = end;
        for( int i=_kids.length-1; i>=0; i-- ) {
          int b = _kids[i][idx];
          assert lt.isOrdered(b,x);
          x = b;
        }
      }

      _keys = null;             // No need to pass Keys back
      _begs = null;             // No need to pass begs back
      _ends = null;             // No need to pass ends back
      tryComplete();
    }
    // Combine Key/beg lists from across the cluster
    @Override public void reduce(Split spl) {
      if( spl == null ) return;
      if( _kids == null ) _kids = spl._kids;
      else for( int k=0; k<_kids.length; k++ )
             for( int i=0; i<_kids[k].length; i++ )
               if( spl._kids[k][i] != 0 )
                 _kids[k][i] = spl._kids[k][i];
    }

    public void makeKids( DRFTree drf ) {
      assert drf._kids == null;
      drf._kids = new DRFTree[_kids.length];
      for( int k=0; k<_kids.length; k++ )
        drf._kids[k] = new DRFTree(_kids[k]);
    }
  }

  // ---
  // Split the top-level tree root based on a sampling criteria.  Move the
  // sampled-out rows to one end and the sampled-in rows to the other.
  private static class SampleSplit extends Split<SampleSplit> {
    // INPUTS
    final double _sampleRate;
    final long _seed;
    SampleSplit( Key keys[], int begs[], int[] ends, double sampleRate, long seed ) {
      super(keys,begs,ends);
      _sampleRate = sampleRate;
      _seed = seed;
    }
    @Override void doSplit( LocalTree lt, int beg, int end ) {
      assert beg == 0;          // Sampling is over the whole (local) tree
      long seed = _seed^end; // Determinstic tree seed, reseeded by rows
      Random rand = Utils.getRNG(seed);

      // Count how many in each bin
      int mid=0;
      for( int i=beg; i<end; i++ )
        if( rand.nextFloat() <= _sampleRate )
          mid++;

      // Pass 2: split by bin-selection: re-run the RNG again
      Random rerun = Utils.getRNG(seed);
      long[] rs0 = lt._rows0;
      long[] rs1 = lt._rows1;
      int j=beg, k=mid;
      for( int i=beg; i<end; i++ )
        if( rerun.nextFloat() <= _sampleRate ) rs1[j++] = rs0[i];
        else                                   rs1[k++] = rs0[i];
      assert j==mid && k==end;
      // Performance wart: should/could flip back and forth between rows0 & rows1
      System.arraycopy(rs1,beg,rs0,beg,(end-beg));

      // Fill in the ranges for the split
      int idx = H2O.SELF.index();
      _kids = new int[2][_begs.length];
      _kids[0][idx] = beg;
      _kids[1][idx] = mid;
    }
  }

  // ---
  // Split based on a histogram & column choice
  private static class HistoSplit extends Split<HistoSplit> {
    // INPUT - cleared after local work
    DHisto2 _dh2;               // Histogram to split on
    HistoSplit( Key keys[], int begs[], int[] ends, DHisto2 dh2 ) {
      super(keys,begs,ends);
      if( dh2._nbins < 2 ) throw H2O.unimpl(); // Nothing to split?
      _dh2 = dh2;
    }
    // Split rows from beg to end
    @Override void doSplit( LocalTree lt, int beg, int end ) {
      int nbins = _dh2._nbins;
      int nclass= _dh2._nclass;
      // Get size of each bin, so we can get start of each split...
      int idx = H2O.SELF.index();
      int offs[] = new int[nbins];
      for( int b=0; b<nbins; b++ )
        offs[b] = _dh2._bins[b][idx];
      // Roll up start point for each bin.  Fill in children.
      _kids = new int[nbins][_begs.length];
      int off = beg;
      for( int b=0; b<nbins; b++ ) {
        _kids[b][idx] = off;
        int tmp = offs[b];
        offs[b] = off;
        off += tmp;
      }

      // Split based on min/step/max bins
      Vec v0 = lt._fr._vecs[_dh2._col];
      long[] rs0 = lt._rows0;
      long[] rs1 = lt._rows1;
      for( int i=beg; i<end; i++ ) {
        long r = rs0[i];
        float f = (float)v0.at(r);
        int b = _dh2.bin(f);    // Which bin?  Linear interpolation.
        rs1[offs[b]++] = r;
      }
      assert offs[nbins-1]==end : ""+Arrays.toString(offs)+", end="+end;

      // Performance wart: should/could flip back and forth between rows0 & rows1
      System.arraycopy(rs1,beg,rs0,beg,(end-beg));
      _dh2 = null;              // Do not pass this back
    }
    @Override public boolean logVerbose() { return false; }
  }

  // ---
  // Class for computing a Confusion Matrix in the Out-Of-Bag-Error-Estimate style
  private static class OOBEETask extends DRemoteTask<OOBEETask> {
    Forest _forest;
    Frame _fr;
    final float _sampleRate;
    long _cm[][];               // OOBEE-style Confusion Matrix
    OOBEETask( Forest forest, float sampleRate, Frame fr ) { _forest = forest;  _sampleRate=sampleRate;  _fr=fr;}
    @Override public void lcompute() {
      // Load all the trees locally, init RNGs to replay the sampling
      int ntrees = _forest._treeKeys.length;
      byte tbits[][] = new byte[ntrees][];
      Random rands[] = new Random[ntrees]; // Also make a pile-o-RNGs
      for( int i=0; i<ntrees; i++ ) {
        tbits[i] = DKV.get(_forest._treeKeys[i]).getBytes();
        assert UDP.get4(tbits[i],0/*tree id*/)==i; // Check tree-bits for sane ID's
        rands[i] = Utils.getRNG(UDP.get8(tbits[i],4/*seed*/));
      }

      // Get the response column & class ymin offset
      Vec vecs[] = _fr._vecs;
      Vec vy = vecs[vecs.length-1]; // Response is last vec
      int ymin = (int)vy.min();
      if( ymin != _forest._ymin ) throw H2O.unimpl(); // Need to adjust for alternate ymin
      int ncls = (int)vy.max()-ymin+1;
      if( ncls != _forest._nclass ) throw H2O.unimpl(); // Need to adjust for alternate class count
      int votes[] = new int[ncls]; // Votes on a row
      _cm = new long[ncls][ncls];  // Confusion Matrix

      // Replay the RNG pattern over the rows
      Vec v0 = _fr.firstReadable();
      int nchunks = v0.nChunks();
      for( int chk=0; chk<nchunks; chk++ )
        if( v0.chunkKey(chk).home() ) { // For all local rows
          long end = v0.chunk2StartElem(chk+1);
          for( long r=v0.chunk2StartElem(chk); r<end; r++ ) {
            Arrays.fill(votes,0);                          // Reset voting for a new row
            for( int t=0; t<ntrees; t++ )                  // For all trees
              if( !(rands[t].nextFloat() <= _sampleRate) ) // Replay sampler, negated
                DRFTree.score(tbits[t],_fr,r,votes);       // Get votes for this tree
            // Find max vote.  No tie-breaker logic.
            int m = 0;
            for( int i=1; i<ncls; i++ )
              if( votes[i] > votes[m] ) m=i;
            int y = (int)vy.at8(r)-ymin; // Actual answer
            _cm[y][m]++;                 // We voted class 'm'.
          }
        }

      _forest = null;           // Do not pass this back
      _fr = null;
      tryComplete();
    }
    @Override public void reduce(OOBEETask cm) {
      if( _cm == null ) _cm = cm._cm;
      else if( cm._cm != null )
        for( int i=0; i<_cm.length; i++ )
          for( int j=0; j<_cm.length; j++ )
            _cm[i][j] += cm._cm[i][j];
    }
  }

  // ---
  // Compute a distributed histogram, on a single column, on a range of rows
  // for a single tree.
  private static class DHisto2 extends DRemoteTask<DHisto2> {
    public static final int BINS=4;
    // INPUTS, cleared when done
    DRFTree _drf;         // Tree-leaf being histogramed
    int[/*node*/] _ends;  // Endpoints for this leaf, per jvm
    final int _col;       // Column being histogramed
    final char _nbins;          // Number of bins
    final short _nclass;        // Number of classes (or 0 for regression)
    final float _min, _step;    // Column min/step, used for binning
    // OUTPUTS, set locally, rolled-up
    float _mins[], _maxs[];     // min/max per-bin across the cluster
    int _clss[/*bin*/][/*class*/]; // Class distribution, rolled-up across the cluster
    int _bins[/*bin*/][/*node*/];  // Local (per-node) bin sizes

    // Build a basic histogram shell
    DHisto2( DRFTree drf, int[] ends, short nclass, int col, float min, float max, boolean isInt, String name ) {
      assert drf._keys != null; // Have keys
      assert drf._kids == null; // No kids yet
      assert drf.check(ends);   // Quicky bounds check
      assert max > min : "Caller ensures "+max+">"+min+", since if max==min the column "+name+" is all constants";
      int nelems=0;             // Count elements across the cluster in this split
      for( int i=0; i<ends.length; i++ )
        nelems += ends[i]-drf._begs[i];
      assert nelems > 0;        // Something to histogram
      _drf  = drf;              // Tree node; LocalTree keys & beg points of row
      _ends = ends;             // Endpoints of rows
      _col  = col;              // Column being histogram

      int xbins = Math.max(Math.min(BINS,nelems),1); // Default bin count
      // See if we can show there are fewer unique elements than nbins.
      // Common for e.g. boolean columns, or near leaves.
      _nbins = (char)((isInt && max-min < xbins)
                      ? ((long)max-(long)min+1L) // Shrink bins
                      : xbins); // Default size for most columns
      _nclass = nclass;
      _min=min;
      _step = (max-min)/_nbins; // Step size for linear interpolation
    }

    @Override public void lcompute() {
      int idx = H2O.SELF.index();
      LocalTree lt = DKV.get(_drf._keys[idx]).get();
      int beg = _drf._begs[idx];
      int end =      _ends[idx];
      _mins = new float[_nbins];  Arrays.fill(_mins, Float.MAX_VALUE);
      _maxs = new float[_nbins];  Arrays.fill(_maxs,-Float.MAX_VALUE);
      _clss = new int  [_nbins][_nclass];
      assert lt.isOrdered(beg,end);
      Vec v0 = lt._fr._vecs[_col];
      Vec vy = lt._fr._vecs[lt._fr.numCols()-1]; // Response is last vec
      int ymin = (int)vy.min();

      // Get the chunks we're working out of
      if( end > beg ) {
        long r0 = lt._rows0[beg];
        Chunk c0 = v0.chunk(r0);
        Chunk cy = vy.chunk(r0);
        assert c0._start==cy._start;
        assert c0._len  ==cy._len  ;
        long cbeg = c0._start;
        long cend = cbeg+c0._len;

        // Visit all rows, chunk by chunk
        for( int i=beg; i<end; i++ ) {
          long r = lt._rows0[i];
          if( r >= cend ) {
            c0 = v0.chunk(r);
            cy = vy.chunk(r);
            cbeg = c0._start;
            cend = cbeg+c0._len;
          }
          float f = (float)c0.at0 ((int)(r-cbeg));
          int y   = (int)  cy.at80((int)(r-cbeg));
          int b = bin(f);         // Which bin?
          if( f < _mins[b] ) _mins[b] = f;
          if( f > _maxs[b] ) _maxs[b] = f;
          _clss[b][y-ymin]++;     // Bump class histogram
        }
      }

      // Roll-up class-counts to bin-counts for this node
      _bins = new int[_nbins][H2O.CLOUD.size()];
      for( int b=0; b<_nbins; b++ ) {
        int sum=0;
        for( int c=0; c<_nclass; c++ ) sum += _clss[b][c];
        _bins[b][idx] = sum;
      }

      _drf = null;              // No need to pass DRFTree back
      _ends = null;             // No need to pass ends back
      tryComplete();
    }
    // Combine min/max/class distributions from across the cluster
    @Override public void reduce(DHisto2 dh2) {
      if( dh2 == null ) return;
      if( _mins==null ) copyOver(dh2);
      else
        for( int i=0; i<_nbins; i++ ) {
          if( _mins[i] > dh2._mins[i] ) _mins[i] = dh2._mins[i];
          if( _maxs[i] < dh2._maxs[i] ) _maxs[i] = dh2._maxs[i];
          for( int c=0; c<_nclass; c++ ) // Merge class distributions
            _clss[i][c] += dh2._clss[i][c];
          for( int j=0; j<H2O.CLOUD.size(); j++ )
            _bins[i][j] += dh2._bins[i][j];
        }
    }

    // Convert a column value to a bin with simple linear interpolation
    int bin( float d ) {
      int idx1  = _step <= 0.0 ? 0 : (int)((d-_min)/_step);
      int idx2  = Math.max(Math.min(idx1,_nbins-1),0); // saturate at bounds
      return idx2;
    }
    // min/max from the histogram, skipping empty bins
    float min() {
      int mb=0;
      while( _mins[mb] ==  Float.MAX_VALUE ) mb++;
      return _mins[mb];
    }
    float max() {
      int xb=_nbins-1;
      while( _maxs[xb] == -Float.MAX_VALUE ) xb--;
      return _maxs[xb];
    }

    // After a histogram has been built, we can check to see if a split (or any
    // following split) based on this histogram has any predictive power.  If
    // min==max then the predictors are not helpful.  If some bins are empty,
    // then again this does not help.  We will stop tracking these splits.
    boolean constantPredictor() { return min() == max();  }

    // Finally, if the response-variable is the same for the entire split,
    // again splitting this column will not help.  Note that if the
    // response-variable is the same for the whole split, we get zero error.
    boolean constantResponse() {
      int b=0;
      while( _mins[b] ==  Float.MAX_VALUE ) b++;
      // Find some class bit somewhere in the non-empty bin mb
      int cidx=0;
      while( _clss[b][cidx] == 0 ) cidx++;
      // See if we got class bits in any other class bin
      for( ; b < _nbins; b++ )
        for( int c=0; c<_nclass; c++ )
          if( _clss[b][c] > 0 && c != cidx ) return false;
      return true;
    }

    // Compute a "score" for a column; lower score "wins" (is a better split).
    // Score for a Classification tree is sum of the errors per-class.  We
    // predict a row using the ratio of the response class to all the rows in
    // the split.
    //
    // Example: we have 10 rows, classed as 8 C0's, 1 C1's and 1 C2's.  The
    // C0's are all predicted as "80% C0".  We have 8 rows which are 80%
    // correctly C0, and 2 rows which are 10% correct.  The total error is
    // 8*(1-.8)+2*(1-.1) = 3.4.
    //
    // Example: we have 10 rows, classed as 6 C0's, 2 C1's and 2 C2's.  Total
    // error is: 6*(1-.6)+4*(1-.2) = 5.6
    float scoreClassification( ) {
      float sum = 0;
      for( int b=0; b<_nbins; b++ ) {
        // A little algebra, and the math we need is:
        //    N - (sum(clss^2)/N)
        long cnt=0, err=0;
        for( int j=0; j<_nclass; j++ ) {
          long c = _clss[b][j];
          cnt += c;
          err += c*c;
        }
        if( cnt > 0 )
          sum += (float)cnt - ((float)err/cnt);
      }
      return sum;
    }
    @Override public boolean logVerbose() { return false; }
  }

  // --------------------------------------------------------------------------
  // The local top-level tree-info for one tree.
  private static class LocalTree extends Iced {
    final Key _key;             // A unique cookie for this datastructure
    final Frame _fr;            // Frame for data-layout and handy vec access
    final long [] _rows0;       // All the local datarows
    final long [] _rows1;       // Double-buffered for easier splitting
    LocalTree( Key key, Frame fr ) {
      _key = key;
      _fr = fr;
      long sum=0;
      Vec v0 = fr.firstReadable();
      int nchunks = v0.nChunks();
      for( int i=0; i<nchunks; i++ )
        if( v0.chunkKey(i).home() )
          sum += v0.chunk2StartElem(i+1)-v0.chunk2StartElem(i);
      if( sum >= Integer.MAX_VALUE ) throw H2O.unimpl(); // More than 2bil rows per node?

      // Get space for 'em
      _rows0 = MemoryManager.malloc8((int)sum);
      _rows1 = MemoryManager.malloc8((int)sum);

      // Fill in the rows this Node will work on
      int k=0;
      for( int i=0; i<nchunks; i++ )
        if( v0.chunkKey(i).home() ) {
          long end = v0.chunk2StartElem(i+1);
          for( long j=v0.chunk2StartElem(i); j<end; j++ )
            _rows0[k++] = j;
      }
      assert isOrdered(0,_rows0.length);
    }

    private boolean isOrdered( int beg, int end ) {
      if( beg == end ) return true;
      long r = _rows0[beg];
      for( int i = beg+1; i<end; i++ ) {
        long s = _rows0[i];
        if( r > s ) {
          System.out.println("Out-Of-Order beg="+beg+" : i-1,i="+i+" : end="+end+", ("+r+" <= "+s+")");
          return false;
        }
        r = s;
      }
      return true;
    }
  }

  static void swap( byte  bs[], int i, int j ) { byte  tmp = bs[i]; bs[i] = bs[j]; bs[j] = tmp; }
  static void swap( int   is[], int i, int j ) { int   tmp = is[i]; is[i] = is[j]; is[j] = tmp; }
  static void swap( float fs[], int i, int j ) { float tmp = fs[i]; fs[i] = fs[j]; fs[j] = tmp; }
  static void swap( long  ls[], int i, int j ) { long  tmp = ls[i]; ls[i] = ls[j]; ls[j] = tmp; }
}
