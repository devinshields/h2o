package hex.gbm;

import hex.gbm.DTree.*;
import hex.rng.MersenneTwisterRNG;
import java.util.Arrays;
import java.util.Random;
import water.*;
import water.H2O.H2OCountedCompleter;
import water.fvec.*;
import water.util.Log.Tag.Sys;
import water.util.Log;

// Random Forest Trees
public class DRF extends Job {
  public static final String KEY_PREFIX = "__DRFModel_";

  // The Key of Trees
  public Key _treeKey;

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
    short nclass = vresponse._isInt ? (short)(vresponse.max()-ymin+1) : 0;

    // Make a master list of trees to be worked on.
    // Make a deterministic set of RNG seeds used to pick split columns.
    final Trees trees = new Trees(ntrees,new MersenneTwisterRNG(new int[]{(int)(seed>>32L),(int)seed}));
    _treeKey = Key.make(Key.make().toString(),(byte)0,Key.BUILT_IN_KEY);
    UKV.put(_treeKey,trees);

    // Get Nodes busy making trees.
    new DRemoteTask() {
      @Override public void lcompute() {
        int tree;
        while( (tree = ((GetOne)new GetOne().invoke(_treeKey))._tree) != -1 )
          doTree(fr,maxDepth,mtrys,(float)sampleRate,trees._seeds[tree],tree);
        tryComplete();
      }
      @Override public void reduce(DRemoteTask drt) { }
    }.invokeOnAllNodes();

    // Get the updated list
    System.out.println("All trees done! "+t_drf);
    Trees trees2 = UKV.get(_treeKey);
    UKV.remove(_treeKey);
    System.out.println(trees2);
  }

  // Master list of tree-seeds, and which trees are being worked on.
  // This list is atomically updated by nodes as they work on trees.
  private static class Trees extends Iced {
    final long [] _seeds;       // Random gen seeds per-tree
    final Key [] _treeKeys;     // Tree Keys, set as trees are completed
    int _next;                  // Next un-owned tree-gen
    Trees( int ntrees, Random rand ) {
      _treeKeys = new Key[ntrees];
      _seeds = new long[ntrees];
      for( int i=0; i<ntrees; i++ )
        _seeds[i] = rand.nextLong();
    }
  }

  // Atomically fetch a tree# to work on & claim it, or return -1.
  private static class GetOne extends TAtomic<Trees> {
    int _tree = -1;
    @Override public Trees atomic( Trees old ) {
      if( old._next >= old._seeds.length ) return null;
      _tree = old._next++;
      return old;
    }
  }

  // --------------------------------------------------------------------------
  // Compute a single RF tree
  private void doTree( final Frame fr, final int maxDepth, final int mtrys, final double sampleRate, long seed, int tree) {
    Timer t_tree = new Timer();
    System.out.println("Working on tree "+tree);

    // Ask all Nodes to setup for building this one tree, and hand me back a
    // cookie (Key) to guide further tree building.
    DRFTree root = new Start1Tree(fr).invokeOnAllNodes()._drf;
    // Capture and reset root info
    final Key[] keys = root._keys;   root._keys = null;
    final int[] ends = root._begs;   root._begs = new int[ends.length];
    System.out.println("ROOT="+root.toString(ends));

    // Split the top-level tree based on random sampling.
    // root._kids[0] will be sampled-in, and root._kids[1] will be sampled-out
    new SampleSplit(root,keys,ends,sampleRate,seed).invokeOnAllNodes();
    DRFTree root0 = root._kids[0];
    DRFTree root1 = root._kids[1];
    root0._keys = keys;
    // End of tree 'root0' is also the beginning of tree 'root1'.
    final int[] ends0 = root1._begs;
    System.out.println("ROOT2 IN ="+root0.toString(ends0));
    System.out.println("ROOT2 OUT="+root1.toString(ends));

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
    Random rand = new MersenneTwisterRNG(new int[]{(int)(seed2>>32L),(int)seed2});

    // build a tree
    root0.doSplit(fr,ends0,cols,mins,maxs,rand,mtrys,maxDepth);

    // Remove local temp tree storage
    Futures fs = new Futures();
    for( Key k : keys )
      UKV.remove(k,fs);
    fs.blockForPending();
    System.out.println("Done on tree "+tree+" in "+t_tree);
  }

  // The simple tree-node class.  Every tree node has a list of local-rows that
  // landed on that tree node; the list is large and kept in the LocalTree
  // structure on each JVM.  We only keep the start & end region marker in the
  // large LocalTree here.
  private static class DRFTree extends Iced {
    Key [] _keys;               // Keys for local data for this one tree
    int [] _begs;               // Beginning-row for this tree leaf, for all nodes
    DRFTree[] _kids;            // Kids, if any.

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

    // Called to make a new interior DRFTree.  Just a row start.
    DRFTree( int beg ) { 
      _begs = new int[H2O.CLOUD.size()];
      int idx = H2O.SELF.index();
      _begs[idx] = beg;
    }
    // Merge trees on interior DRFTree nodes.
    // Merge all the kid-tree starts
    void add2( DRFTree drf ) {
      if( drf._kids == null ) return;
      if( _kids == null ) _kids = drf._kids;
      else for( int i=0; i<_kids.length; i++ )
             _kids[i].add3(drf._kids[i]);
    }
    void add3( DRFTree drf ) {
      assert _kids==null;
      for( int i=0; i<_begs.length; i++ )
        if( drf._begs[i] != 0 )
          _begs[i] = drf._begs[i];
    }

    // ---
    // Build a tree level.  Passed in a set of active columns, and their min/max's.
    // Conceptually, repeat for mtrys columns:
    // (0) Figure a column to split on, drawn at random without replacement.
    // (1) build histograms of those columns
    // (2) figure out best split
    // (3) split rows
    // Then (4) recursively compute a new level by doing the above steps on
    // each of the several splits.
    void doSplit(Frame fr, int ends[], int cols[], float mins[], float maxs[], Random rand, int mtrys, int depth) {
      if( depth == 0 ) return;
      // We pick up to mtrys columns, pulling at random without replacement
      // from the entire list of columns.  Picked columns are moved to the end
      // of the column list, and are not picked again.
      int res_col=cols.length;
      for( int i=0; i<mtrys; i++ ) {
        if( res_col == 0 ) break; // Out of choices!
        // (0) Figure a column to split on, drawn at random without replacement.
        int col = rand.nextInt(res_col);
        // Move selected column to the end, so we do not pick it again
        int   tmp1 = cols[col]; cols[col] = cols[--res_col];  cols[res_col] = tmp1;
        float tmp2 = mins[col]; mins[col] = mins[  res_col];  mins[res_col] = tmp2;
        float tmp3 = maxs[col]; maxs[col] = maxs[  res_col];  maxs[res_col] = tmp3;

        // (1) build histograms
        System.out.println("Score col "+tmp1+", min="+tmp2+", max="+tmp3);
        DHisto2 dh2 = new DHisto2(this,ends,tmp1,tmp2,tmp3,fr._vecs[tmp1]._isInt,fr._names[tmp1]).invokeOnAllNodes();
        System.out.println("mins="+Arrays.toString(dh2._mins)+", maxs="+Arrays.toString(dh2._maxs));
      }


      // (2) figure out best split
      // (3) split rows
      // (4) recursively compute new level
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
      String s = "{";
      for( int i=0; i<_begs.length; i++ )
        s += "["+H2O.CLOUD._memary[i]+","+_begs[i]+"],";
      return s+"}";
    }
    // Pretty-print with the other ends
    public String toString( int[] ends ) {
      String s = "{";
      for( int i=0; i<_begs.length; i++ )
        s += "["+H2O.CLOUD._memary[i]+","+_begs[i]+","+ends[i]+"],";
      return s+"}";
    }
  }

  // ---
  // Start this one Tree on all Nodes.  Return a collection of Keys that
  // uniquely identifies the tree being built on all Nodes.
  private static class Start1Tree extends DRemoteTask<Start1Tree> {
    final Frame _fr;
    DRFTree _drf;
    Start1Tree(Frame fr) { _fr = fr; }
    @Override public void lcompute() {
      // One Key per Tree, homed to self.
      Key myKey = Key.make("T"+Key.make(),(byte)0,Key.BUILT_IN_KEY,H2O.SELF);
      // Stash away local tree info on this Node, awaiting the next pass.
      LocalTree lt = new LocalTree(myKey,_fr);
      UKV.put(myKey,lt);
      _drf = new DRFTree(myKey,lt._rows.length);
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
  private static abstract class Split extends DRemoteTask<Split> {
    DRFTree _drf;               // The guy being split
    int _ends[];                // Ends of split rows
    abstract void doSplit(LocalTree lt, int beg, int end);
    Split( DRFTree drf, Key[] keys, int[] ends ) { 
      assert drf._keys == null; // These where cleaned out last pass
      assert drf._kids == null; // No kids yet
      assert drf.check(ends);   // Quicky bounds check
      drf._keys = keys;         // Set in afresh
      _drf  = drf; 
      _ends = ends;
    }
    @Override public void lcompute() {
      int idx = H2O.SELF.index();
      LocalTree lt = DKV.get(_drf._keys[idx]).get();
      int beg = _drf._begs[idx];
      int end =      _ends[idx];
      assert lt.isOrdered(beg,end) : "Not ordered";
      doSplit(lt,beg,end);
      assert lt.isOrdered(beg,end) : "Not ordered";
      _drf._keys = null;        // No need to pass Keys back
      _drf._begs = null;        // No need to pass begs back
      _ends = null;             // No need to pass ends back
      tryComplete();
    }
    // Combine Key/beg lists from across the cluster
    @Override public void reduce(Split spl) {
      if( spl == null ) return;
      if( _drf == null ) _drf = spl._drf;
      else _drf.add2(spl._drf);
    }
  }

  // ---
  // Split the top-level tree root based on a sampling criteria.  Move the
  // sampled-out rows to one end and the sampled-in rows to the other.
  private static class SampleSplit extends Split {
    final double _sampleRate;
    final long _seed;
    SampleSplit( DRFTree root, Key[] keys, int[] ends, double sampleRate, long seed ) { 
      super(root,keys,ends); 
      _sampleRate = sampleRate;
      _seed = seed;
    }
    void doSplit( LocalTree lt, int beg, int end ) {
      assert beg == 0;          // Sampling is over the whole (local) tree
      long seed = _seed^end; // Determinstic tree seed, reseeded by rows
      Random rand = new MersenneTwisterRNG(new int[]{(int)(seed>>32L),(int)seed});

      // Sample over all rows, splitting the rows
      int j=end;
      for( int i=beg; i<j; i++ )
        if( !(rand.nextFloat() <= _sampleRate) ) // Sample this row OUT
          lt.swap(i--,--j);
      Arrays.sort(lt._rows,beg,end);

      _drf._kids    = new DRFTree[2];
      _drf._kids[0] = new DRFTree(beg);
      _drf._kids[1] = new DRFTree( j );
    }
  }

  // ---
  // Compute a distributed histogram, on a single column, on a range of rows
  // for a single tree.
  private static class DHisto2 extends DRemoteTask<DHisto2> {
    public static final int BINS=4;
    DRFTree _drf;         // Tree-leaf being histogramed
    int[]   _ends;        // Endpoints for this leaf, per jvm
    final int _col;       // Column being histogramed
    final char _nbins;          // Number of bins 
    final float _min, _step;    // Column min/step, used for binning
    float _mins[], _maxs[];     // min/max per-bin across the cluster

    // Build a basic histogram shell
    DHisto2( DRFTree drf, int[] ends, int col, float min, float max, boolean isInt, String name ) {
      assert drf._keys != null; // Have keys
      assert drf._kids == null; // No kids yet
      assert drf.check(ends);   // Quicky bounds check
      assert max > min : "Caller ensures "+max+">"+min+", since if max==min== the column "+name+" is all constants";
      int nelems=0;             // Count elements across the cluster in this split
      for( int i=0; i<ends.length; i++ )
        nelems += ends[i]-drf._begs[i];
      assert nelems > 0;        // Something to histogram
      _ends = ends;             // Endpoints of rows
      _col = col;               // Column being histogram

      int xbins = Math.max(Math.min(BINS,nelems),1); // Default bin count
      // See if we can show there are fewer unique elements than nbins.
      // Common for e.g. boolean columns, or near leaves.
      _nbins = (char)((isInt && max-min < xbins) 
                      ? ((long)max-(long)min+1L) // Shrink bins
                      : xbins); // Default size for most columns        
      _min=min; 
      _step = (max-min)/_nbins; // Step size for linear interpolation
      System.out.println("bins="+(int)_nbins);
    }

    @Override public void lcompute() {
      System.out.println("local compute bins="+(int)_nbins);
      int idx = H2O.SELF.index();
      LocalTree lt = DKV.get(_drf._keys[idx]).get();
      int beg = _drf._begs[idx];
      int end =      _ends[idx];
      _mins = new float[_nbins];  Arrays.fill(_mins, Float.MAX_VALUE);
      _maxs = new float[_nbins];  Arrays.fill(_mins,-Float.MAX_VALUE);
      assert lt.isOrdered(beg,end);
      Vec v0 = lt._fr._vecs[_col];

      for( int i=beg; i<end; i++ ) {
        float f = (float)v0.at(lt._rows[i]);
        int b = bin(f);         // Which bin?
        if( f < _mins[b] ) _mins[b] = f;
        if( f > _maxs[b] ) _maxs[b] = f;
      }

      _drf = null;              // No need to pass DRFTree back
      _ends = null;             // No need to pass ends back
      tryComplete();
    }
    // Combine min/max/class distributions from across the cluster
    @Override public void reduce(DHisto2 dh2) {
      System.out.println("reduce, mins="+Arrays.toString(_mins));
      if( dh2 == null ) return;
      for( int i=0; i<_nbins; i++ ) {
        if( _mins[i] > dh2._mins[i] ) _mins[i] = dh2._mins[i];
        if( _maxs[i] < dh2._maxs[i] ) _maxs[i] = dh2._maxs[i];
      }
    }

    int bin( float d ) {
      int idx1  = _step <= 0.0 ? 0 : (int)((d-_min)/_step);
      int idx2  = Math.max(Math.min(idx1,_nbins-1),0); // saturate at bounds
      return idx2;
    }
  }


  // --------------------------------------------------------------------------
  // The local top-level tree-info for one tree.
  private static class LocalTree extends Iced {
    final Key _key;             // A unique cookie for this datastructure
    final Frame _fr;            // Frame for the tree being built
    final long [] _rows;        // All the local datarows
    LocalTree( Key key, Frame fr ) {
      _key = key;
      _fr = fr;
      Vec v0 = fr.firstReadable();
      long sum=0;
      int nchunks = v0.nChunks();
      for( int i=0; i<nchunks; i++ )
        if( v0.chunkKey(i).home() )
          sum += v0.chunk2StartElem(i+1)-v0.chunk2StartElem(i);
      if( sum >= Integer.MAX_VALUE ) throw H2O.unimpl(); // More than 2bil rows per node?
      
      // Get space for 'em
      _rows = MemoryManager.malloc8((int)sum);
      
      // Fill in the rows this Node will work on
      int k=0;
      for( int i=0; i<nchunks; i++ )
        if( v0.chunkKey(i).home() ) {
          long end = v0.chunk2StartElem(i+1);
          for( long j=v0.chunk2StartElem(i); j<end; j++ )
            _rows[k++] = j;
      }    
      assert isOrdered(0,_rows.length);
    }
    void swap( int i, int j ) { long tmp = _rows[i]; _rows[i] = _rows[j]; _rows[j] = tmp; }

    private boolean isOrdered( int beg, int end ) {
      if( beg == end ) return true;
      long r = _rows[beg];
      for( int i = beg+1; i<end; i++ ) {
        long s = _rows[i];
        if( r > s ) return false;
        r = s;
      }
      return true;
    }
  }
}
