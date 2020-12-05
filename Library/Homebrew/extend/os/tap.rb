# typed: strict
# frozen_string_literal: true

require "extend/os/linux/tap" if OS.linux?
require "extend/os/mac/tap" if OS.mac?
