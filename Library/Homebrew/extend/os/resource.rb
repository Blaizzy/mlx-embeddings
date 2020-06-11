# frozen_string_literal: true

if OS.mac?
  require "extend/os/mac/resource"
elsif OS.linux?
  require "extend/os/linux/resource"
end
