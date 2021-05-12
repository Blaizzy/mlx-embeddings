# typed: true
# frozen_string_literal: true

require "pathname"

HOMEBREW_LIBRARY_PATH = Pathname(__dir__).parent.realpath.freeze

require_relative "../utils/gems"
Homebrew.setup_gem_environment!(setup_path: false)

$LOAD_PATH.push HOMEBREW_LIBRARY_PATH.to_s unless $LOAD_PATH.include?(HOMEBREW_LIBRARY_PATH.to_s)
require_relative "../vendor/bundle/bundler/setup"
$LOAD_PATH.uniq!

# Block any gem loading by bypassing rubygem's `require`.
# Helps make sure we don't accidentally use things not in bundler's load path.
# Bundler 2.2.7+ and non-standalone mode both do this automatically.
# https://github.com/rubygems/rubygems/blob/5841761974bef324a33ef1cb650bbf8a2457805b/bundler/lib/bundler/installer/standalone.rb#L55-L63
if Kernel.private_method_defined?(:gem_original_require)
  Kernel.send(:remove_method, :require)
  Kernel.send(:define_method, :require, Kernel.instance_method(:gem_original_require))
  Kernel.send(:private, :require)
end
