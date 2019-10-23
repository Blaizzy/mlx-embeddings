#!/usr/bin/swift

import Foundation

extension FileHandle : TextOutputStream {
  public func write(_ string: String) {
    self.write(string.data(using: .utf8)!)
  }
}

var stderr = FileHandle.standardError

let manager: FileManager = FileManager()

var success = true

for item in CommandLine.arguments[1...] {
  do {
    let path: URL = URL(fileURLWithPath: item)
    var trashedPath: NSURL!
    try manager.trashItem(at: path, resultingItemURL: &trashedPath)
    print((trashedPath as URL).path, terminator: ":")
  } catch {
    print(item, terminator: ":", to: &stderr)
    success = false
  }
}

guard success else {
  exit(1)
}
