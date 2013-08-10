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
          doTree(fr,maxDepth,ntrees,mtrys,(float)sampleRate,trees._seeds[tree],tree);
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
  private void doTree( final Frame fr, final int maxDepth, final int ntrees, final int mtrys, final double sampleRate, long seed, int tree) {
    Timer t_tree = new Timer();
    System.out.println("Working on tree "+tree);

    // Build the local set of rows.  We build histograms from local tree data,
    // then roll up the histograms globally.  When building a histogram, we'll
    // keep all the rows of a split adjacent in this rows array, and sorted.
    long[] rows = buildLocalRows(fr.firstReadable());

    // build a tree-node 
    

    System.out.println("Done on tree "+tree+" in "+t_tree);
  }

  // --------------------------------------------------------------------------
  private long[] buildLocalRows(Vec v0) {
    long sum=0;
    int nchunks = v0.nChunks();
    for( int i=0; i<nchunks; i++ )
      if( v0.chunkKey(i).home() )
        sum += v0.chunk2StartElem(i+1)-v0.chunk2StartElem(i);
    System.out.println("Found "+sum+" local rows ");
    if( sum >= Integer.MAX_VALUE ) throw H2O.unimpl(); // More than 2bil rows per node?

    // Get space for 'em
    long[] rows = MemoryManager.malloc8((int)sum);

    // Fill in the rows this Node will work on
    int k=0;
    for( int i=0; i<nchunks; i++ )
      if( v0.chunkKey(i).home() ) {
        long end = v0.chunk2StartElem(i+1);
        for( long j=v0.chunk2StartElem(i); j<end; j++ )
          rows[k++] = j;
      }    
    
    return rows;
  }

}
