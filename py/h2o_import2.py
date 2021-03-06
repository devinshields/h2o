import h2o, h2o_cmd, h2o_jobs
import time, re, getpass
import os

# hdfs/maprfs/s3/s3n paths should be absolute from the bucket (top level)
# so only walk around for local
def find_folder_path_and_pattern(bucket, pathWithRegex):

    # strip the common mistake of leading "/" in path, if bucket is specified too
    print "pathWithRegex:", pathWithRegex
    if re.match("/", pathWithRegex):
        print "You said bucket:", bucket, "so stripping incorrect leading '/' from", pathWithRegex
        pathWithRegex = pathWithRegex.lstrip('/')

    if bucket is None:  # good for absolute path name
        bucketPath = ""

    elif bucket == ".":
        bucketPath = os.getcwd()

    # does it work to use bucket "." to get current directory
    elif os.environ.get('H2O_BUCKETS_ROOT'):
        h2oBucketsRoot = os.environ.get('H2O_BUCKETS_ROOT')
        print "Using H2O_BUCKETS_ROOT environment variable:", h2oBucketsRoot

        rootPath = os.path.abspath(h2oBucketsRoot)
        if not (os.path.exists(rootPath)):
            raise Exception("H2O_BUCKETS_ROOT in env but %s doesn't exist." % rootPath)

        bucketPath = os.path.join(rootPath, bucket)
        if not (os.path.exists(bucketPath)):
            raise Exception("H2O_BUCKETS_ROOT and path used to form %s which doesn't exist." % bucketPath)

    else:
        (head, tail) = os.path.split(os.path.abspath(bucket))
        print "find_bucket looking upwards from", head, "for", tail
        # don't spin forever 
        levels = 0
        while not (os.path.exists(os.path.join(head, tail))):
            print "Didn't find", tail, "at", head
            head = os.path.split(head)[0]
            levels += 1
            if (levels==10):
                raise Exception("unable to find bucket: %s" % bucket)

        print "Did find", tail, "at", head
        bucketPath = os.path.join(head, tail)

    # if there's no path, just return the bucketPath
    # but what about cases with a header in the folder too? (not putfile)
    if pathWithRegex is None:
        return (bucketPath, None)

    # if there is a "/" in the path, that means it's not just a pattern
    # split it
    # otherwise it is a pattern. use it to search for files in python first? 
    # FIX! do that later
    elif "/" in pathWithRegex:
        (head, tail) = os.path.split(pathWithRegex)
        folderPath = os.path.join(bucketPath, head)
        if not os.path.exists(folderPath):
            raise Exception("%s doesn't exist. %s under %s may be wrong?" % (folderPath, head, bucketPath))
    else:
        folderPath = bucketPath
        tail = pathWithRegex
        
    print "folderPath:", folderPath, "tail:", tail
    return (folderPath, tail)


# passes additional params thru kwargs for parse
# use_header_file
# header
# exclude
# src_key can be a pattern
# can import with path= a folder or just one file
def import_only(node=None, schema="put", bucket=None, path=None,
    timeoutSecs=30, retryDelaySecs=0.5, initialDelaySecs=0.5, pollTimeoutSecs=180, noise=None,
    noPoll=False, doSummary=True, src_key='python_src_key', **kwargs):

    # no bucket is sometimes legal (fixed path)
    if not node: node = h2o.nodes[0]

    if "/" in path:
        (head, pattern) = os.path.split(path)
    else:
        (head, pattern)  = ("", path)

    if schema=='put':
        if not path: raise Exception('path=, No file to putfile')
        (folderPath, filename) = find_folder_path_and_pattern(bucket, path)
        print "folderPath:", folderPath, "filename:", filename
        filePath = os.path.join(folderPath, filename)
        print 'filePath:', filePath
        key = node.put_file(filePath, key=src_key, timeoutSecs=timeoutSecs)
        return (None, key)

    elif schema=='s3' or node.redirect_import_folder_to_s3_path:
        folderURI = "s3://" + bucket + "/" + head
        importResult = node.import_s3(bucket, timeoutSecs=timeoutSecs)

    elif schema=='s3n':
        folderURI = "s3n://" + bucket + "/" + head
        importResult = node.import_hdfs(folderURI, timeoutSecs=timeoutSecs)

    elif schema=='maprfs':
        folderURI = "maprfs://" + bucket + "/" + head
        importResult = node.import_hdfs(folderURI, timeoutSecs=timeoutSecs)

    elif schema=='hdfs' or node.redirect_import_folder_to_s3n_path:
        folderURI = "hdfs://" + node.hdfs_name_node + "/" + bucket + "/" + head
        importResult = node.import_hdfs(folderURI, timeoutSecs=timeoutSecs)

    elif schema=='local':
        (folderPath, pattern) = find_folder_path_and_pattern(bucket, path)
        folderURI = 'nfs:/' + folderPath
        importResult = node.import_files(folderPath, timeoutSecs=timeoutSecs)

    importPattern = folderURI + "/" + pattern
    return (importResult, importPattern)

# can take header, header_from_file, exclude params
def parse(node=None, pattern=None, hex_key=None,
    timeoutSecs=30, retryDelaySecs=0.5, initialDelaySecs=0.5, pollTimeoutSecs=180, noise=None,
    noPoll=False, **kwargs):

    if not node: node = h2o.nodes[0]

    parseResult = node.parse(node, pattern, hex_key,
        timeoutSecs, retryDelaySecs, initialDelaySecs, pollTimeoutSecs, noise,
        noPoll, **kwargs)

    parseResult['python_source'] = pattern
    return parseResult


def import_parse(node=None, schema="put", bucket=None, path=None,
    src_key=None, hex_key=None, 
    timeoutSecs=30, retryDelaySecs=0.5, initialDelaySecs=0.5, pollTimeoutSecs=180, noise=None,
    noPoll=False, doSummary=False, **kwargs):

    if not node: node = h2o.nodes[0]

    (importResult, importPattern) = import_only(node, schema, bucket, path,
        timeoutSecs, retryDelaySecs, initialDelaySecs, pollTimeoutSecs, noise, noPoll, **kwargs)

    print "importPattern:", importPattern
    print "importResult", h2o.dump_json(importResult)

    parseResult = node.parse(importPattern, hex_key,
        timeoutSecs, retryDelaySecs, initialDelaySecs, pollTimeoutSecs, noise, noPoll, **kwargs)
    print "parseResult:", h2o.dump_json(parseResult)

    # do SummaryPage here too, just to get some coverage
    if doSummary:
        node.summary_page(myKey2, timeoutSecs=timeoutSecs)

    return parseResult
