#!/usr/bin/swift

import Foundation

struct SwiftErr: TextOutputStream {
    public static var stream = SwiftErr()

    mutating func write(_ string: String) {
        fputs(string, stderr)
    }
}

guard CommandLine.arguments.count >= 3 else {
    print("Usage: swift copy-xattrs.swift <source> <dest>")
    exit(2)
}

CommandLine.arguments[2].withCString { destPath in
    let destNamesLen = listxattr(destPath, nil, 0, 0)
    if destNamesLen == -1 {
        print("listxattr for destination failed: \(errno)", to: &SwiftErr.stream)
        exit(1)
    }
    let destNamesBuf = UnsafeMutablePointer<Int8>.allocate(capacity: destNamesLen)
    if listxattr(destPath, destNamesBuf, destNamesLen, 0) != destNamesLen {
        print("Attributes changed during system call", to: &SwiftErr.stream)
        exit(1)
    }

    var destNamesIdx = 0
    while destNamesIdx < destNamesLen {
        let attribute = destNamesBuf + destNamesIdx

        if removexattr(destPath, attribute, 0) != 0 {
            print("removexattr for \(String(cString: attribute)) failed: \(errno)", to: &SwiftErr.stream)
            exit(1)
        }

        destNamesIdx += strlen(attribute) + 1
    }
    destNamesBuf.deallocate()

    CommandLine.arguments[1].withCString { sourcePath in
        let sourceNamesLen = listxattr(sourcePath, nil, 0, 0)
        if sourceNamesLen == -1 {
            print("listxattr for source failed: \(errno)", to: &SwiftErr.stream)
            exit(1)
        }
        let sourceNamesBuf = UnsafeMutablePointer<Int8>.allocate(capacity: sourceNamesLen)
        if listxattr(sourcePath, sourceNamesBuf, sourceNamesLen, 0) != sourceNamesLen {
            print("Attributes changed during system call", to: &SwiftErr.stream)
            exit(1)
        }

        var sourceNamesIdx = 0
        while sourceNamesIdx < sourceNamesLen {
            let attribute = sourceNamesBuf + sourceNamesIdx

            let valueLen = getxattr(sourcePath, attribute, nil, 0, 0, 0)
            if valueLen == -1 {
                print("getxattr for \(String(cString: attribute)) failed: \(errno)", to: &SwiftErr.stream)
                exit(1)
            }
            let valueBuf = UnsafeMutablePointer<Int8>.allocate(capacity: valueLen)
            if getxattr(sourcePath, attribute, valueBuf, valueLen, 0, 0) != valueLen {
                print("Attributes changed during system call", to: &SwiftErr.stream)
                exit(1)
            }

            if setxattr(destPath, attribute, valueBuf, valueLen, 0, 0) != 0 {
                print("setxattr for \(String(cString: attribute)) failed: \(errno)", to: &SwiftErr.stream)
                exit(1)
            }

            valueBuf.deallocate()
            sourceNamesIdx += strlen(attribute) + 1
        }
        sourceNamesBuf.deallocate()
    }
}
