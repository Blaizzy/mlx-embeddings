# typed: true
# frozen_string_literal: true

module Cask
  class Download
    undef quarantine

    def quarantine(_path)
      opoo "Cannot quarantine download: No xattr available on linux." if @quarantine
      nil
    end
  end
end
