# typed: true
# frozen_string_literal: true

if OS.mac?
  require "extend/os/mac/default_prefix"
elsif OS.linux?
  require "extend/os/linux/default_prefix"
end
