# typed: strict
# frozen_string_literal: true

require "extend/os/linux/utils/analytics" if OS.linux?
require "extend/os/mac/utils/analytics" if OS.mac?
