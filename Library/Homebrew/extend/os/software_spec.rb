# typed: strict
# frozen_string_literal: true

if OS.linux?
  require "extend/os/linux/software_spec"
elsif OS.mac?
  require "extend/os/mac/software_spec"
end
