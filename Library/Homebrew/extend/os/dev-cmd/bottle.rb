# typed: strict
# frozen_string_literal: true

require "extend/os/linux/dev-cmd/bottle" if OS.linux?

if OS.mac?
  require "extend/os/mac/dev-cmd/bottle"
elsif OS.linux?
  require "extend/os/linux/dev-cmd/bottle"
end
