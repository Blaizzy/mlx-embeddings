# typed: strict
# frozen_string_literal: true

if OS.mac?
  require "extend/os/mac/cmd/update-report"
elsif OS.linux?
  require "extend/os/linux/cmd/update-report"
end
