# typed: strict
# frozen_string_literal: true

if OS.mac?
  require "extend/os/mac/dev-cmd/bottle"
elsif OS.linux?
  require "extend/os/linux/dev-cmd/bottle"
end
