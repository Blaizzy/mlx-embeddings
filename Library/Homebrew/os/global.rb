# typed: strict
# frozen_string_literal: true

if OS.mac?
  require "os/mac/global"
elsif OS.linux?
  require "os/linux/global"
end
