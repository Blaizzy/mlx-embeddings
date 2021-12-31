#!/usr/bin/swift

import Foundation

extension FileHandle : TextOutputStream {
  public func write(_ string: String) {
      if let data = string.data(using: .utf8) { self.write(data) }
  }
}

var stderr = FileHandle.standardError

let manager = FileManager.default

var success = true

// The command line arguments given but without the script's name
let CMDLineArgs = Array(CommandLine.arguments.dropFirst())

for item in CMDLineArgs {
    do {
        let url = URL(fileURLWithPath: item)
        var trashedPath: NSURL!
        try manager.trashItem(at: url, resultingItemURL: &trashedPath)
        print((trashedPath as URL).path, terminator: ":")
        success = true
    } catch {
        print(item, terminator: ":", to: &stderr)
        success = false
    }
}

guard success else {
    exit(1)
}
