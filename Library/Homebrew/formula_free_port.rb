# frozen_string_literal: true

module Homebrew
  module FreePort
    require "socket"

    def free_port
      server = TCPServer.new 0
      _, port, = server.addr
      server.close

      port
    end
  end
end
