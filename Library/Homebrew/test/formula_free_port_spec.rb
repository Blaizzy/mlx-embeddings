# typed: false
# frozen_string_literal: true

require "socket"
require "formula_free_port"

module Homebrew
  describe FreePort do
    include described_class

    describe "#free_port" do
      # IANA suggests user port from 1024 to 49151
      # and dynamic port for 49152 to 65535
      # http://www.iana.org/assignments/port-numbers
      MIN_PORT = 1024
      MAX_PORT = 65535

      it "returns a free TCP/IP port" do
        port = free_port

        expect(port).to be_between(MIN_PORT, MAX_PORT)
        expect { TCPServer.new(port).close }.not_to raise_error
      end
    end
  end
end
